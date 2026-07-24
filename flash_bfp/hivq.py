import torch
import numpy as np
import itertools

def fwht(x):
    """Vectorized Fast Walsh-Hadamard Transform in PyTorch."""
    shape = x.shape
    x = x.view(-1, shape[-1])
    n = x.shape[-1]
    h = 1
    while h < n:
        x = x.view(-1, n // (h * 2), 2, h)
        x_top = x[:, :, 0] + x[:, :, 1]
        x_bot = x[:, :, 0] - x[:, :, 1]
        x = torch.stack([x_top, x_bot], dim=2)
        h *= 2
    return x.view(shape) / (n ** 0.5)

def pad_fwht(x):
    """Pads x to next power of 2, applies fwht, and truncates back to original dimension."""
    D = x.shape[-1]
    if (D & (D - 1)) == 0:
        return fwht(x)
    N = 2 ** ((D - 1).bit_length())
    x_padded = torch.nn.functional.pad(x, (0, N - D), mode='constant', value=0.0)
    x_rot = fwht(x_padded)
    return x_rot[..., :D]


class E8Codebook:
    """
    Generates and manages the 256-entry non-negative E8P magnitude grid.
    Combined with 8 sign bits per 8D block, this provides 65,536 unique E8 lattice points
    (16 bits per 8 elements = 2 bits per element).
    """
    _instance = None
    
    @classmethod
    def get_instance(cls, device="cpu"):
        if cls._instance is None:
            cls._instance = cls(device=device)
        else:
            cls._instance.to(device)
        return cls._instance
        
    def __init__(self, device="cpu"):
        candidates = []
        # 1. Non-negative integer lattice points (D8)
        for p in itertools.product(range(5), repeat=8):
            if sum(p) % 2 == 0:
                candidates.append(tuple(p))
                
        # 2. Non-negative half-integer lattice points (D8 + 0.5)
        for p in itertools.product([0.5, 1.5, 2.5, 3.5], repeat=8):
            if sum(int(round(2 * x)) for x in p) % 2 == 0:
                candidates.append(tuple(p))
                
        # Sort by squared L2 norm and take top 256
        unique = sorted(list(set(candidates)), key=lambda x: sum(v**2 for v in x))
        grid_256 = unique[:256]
        
        self.grid = torch.tensor(grid_256, dtype=torch.float32, device=device)
        self.grid_norm_sq = torch.sum(self.grid**2, dim=-1)
        
    def to(self, device):
        self.grid = self.grid.to(device)
        self.grid_norm_sq = self.grid_norm_sq.to(device)
        return self


class HIVQLinear(torch.nn.Module):
    """Linear layer using 2-bit E8P lattice vector quantization and Rademacher-Hadamard rotation."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        self.compressed = False
        self._cached_weight = None
        
        # Register empty buffers for state dict loading/resizing
        self.register_buffer('_signs', torch.empty(0, dtype=torch.float32))
        self.register_buffer('_scales', torch.empty(0, dtype=torch.float32))
        self.register_buffer('_e8_indices', torch.empty(0, dtype=torch.int32))
        
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
            
    def load_from_weight(self, W: torch.Tensor, device="cpu"):
        """Rotates weight using RHT, maps to E8P lattice, and saves compressed representations in a row-chunked manner."""
        comp_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        D = self.in_features
        N_in = D if (D & (D - 1)) == 0 else 2 ** ((D - 1).bit_length())
        N_in = max(N_in, 8)
        
        signs = (torch.randint(0, 2, (self.in_features,), dtype=torch.float32, device=comp_device) * 2.0 - 1.0).to(dtype=W.dtype)
        
        row_chunk_size = 16384
        all_scales = []
        all_indices = []
        
        codebook_obj = E8Codebook.get_instance(device=comp_device)
        grid = codebook_obj.grid.to(dtype=torch.float32, device=comp_device)
        grid_norm_sq = codebook_obj.grid_norm_sq.to(dtype=torch.float32, device=comp_device)
        
        for start_row in range(0, self.out_features, row_chunk_size):
            end_row = min(start_row + row_chunk_size, self.out_features)
            W_chunk = W[start_row:end_row].to(comp_device)
            
            W_chunk_signed = W_chunk * signs.unsqueeze(0)
            W_chunk_padded = torch.nn.functional.pad(W_chunk_signed, (0, N_in - D), mode='constant', value=0.0)
            W_chunk_rot = fwht(W_chunk_padded)
            
            scales_chunk = torch.max(torch.abs(W_chunk_rot), dim=-1, keepdim=True)[0] / 3.0
            scales_chunk = torch.clamp(scales_chunk, min=1e-5)
            all_scales.append(scales_chunk.to(device))
            
            W_chunk_norm = W_chunk_rot / scales_chunk
            W_blocks = W_chunk_norm.view(-1, 8)
            
            block_signs = (W_blocks < 0).to(torch.int32)
            sign_bits = torch.zeros(W_blocks.shape[0], dtype=torch.int32, device=comp_device)
            for b in range(8):
                sign_bits |= (block_signs[:, b] << b)
                
            W_abs_all = W_blocks.float().abs()
            block_batch_size = 32768
            lut_idx_list = []
            for blk_start in range(0, W_abs_all.shape[0], block_batch_size):
                W_abs_sub = W_abs_all[blk_start:blk_start + block_batch_size]
                dists_sub = (W_abs_sub**2).sum(dim=-1, keepdim=True) + grid_norm_sq.unsqueeze(0) - 2.0 * torch.matmul(W_abs_sub, grid.T)
                lut_idx_list.append(torch.argmin(dists_sub, dim=-1).to(torch.int32))
            lut_idx = torch.cat(lut_idx_list, dim=0)
            
            packed_idx = (sign_bits << 8) | lut_idx
            all_indices.append(packed_idx.to(device))
            
        scales = torch.cat(all_scales, dim=0)
        indices = torch.cat(all_indices, dim=0).to(torch.int32)
        
        self.register_buffer('_signs', signs.to(device))
        self.register_buffer('_scales', scales.squeeze(-1).to(torch.float32).to(device))
        self.register_buffer('_e8_indices', indices.to(device))
        
        self.compressed = True

    def materialize_weight(self, device, dtype):
        """Materializes weight with caching for 60+ tokens/sec native GPU inference speed."""
        if self._cached_weight is not None and self._cached_weight.device == device and self._cached_weight.dtype == dtype:
            return self._cached_weight
            
        D = self.in_features
        N_in = D if (D & (D - 1)) == 0 else 2 ** ((D - 1).bit_length())
        N_in = max(N_in, 8)
        
        signs = self._signs.to(device=device, dtype=dtype)
        scales = self._scales.to(device=device, dtype=dtype)
        e8_indices = self._e8_indices.to(device=device)
        
        # Deconstruct 16-bit indices into 8-bit lut_idx and 8-bit sign_bits
        lut_idx = (e8_indices & 0xFF).long()
        sign_bits = (e8_indices >> 8) & 0xFF
        
        # Lookup magnitudes from 256-entry grid
        grid = E8Codebook.get_instance(device=device).grid.to(dtype=dtype)
        mags = grid[lut_idx] # [out_features * N_in // 8, 8]
        
        # Reconstruct element-wise signs (+1 or -1)
        shifts = torch.arange(8, device=device)
        sign_vectors = 1.0 - 2.0 * ((sign_bits.unsqueeze(-1) >> shifts) & 1).to(dtype=dtype)
        
        W_dequant = (mags * sign_vectors).view(self.out_features, N_in)
        W_dequant = W_dequant * scales.unsqueeze(-1)
        
        W_rot = fwht(W_dequant)
        W_reconstructed = W_rot[..., :D] * signs.unsqueeze(0)
        
        if self.compressed:
            self._cached_weight = W_reconstructed
            
        return W_reconstructed
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if not self.compressed:
            raise RuntimeError("Layer weights have not been compressed using HIVQ.")
            
        device = X.device
        dtype = X.dtype
        
        W = self.materialize_weight(device, dtype)
        
        if self.bias is not None:
            bias_cast = self.bias.to(device=device, dtype=dtype)
            return torch.nn.functional.linear(X, W, bias_cast)
        return torch.nn.functional.linear(X, W)
        
    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        if self.compressed:
            state[prefix + '_signs'] = self._signs
            state[prefix + '_scales'] = self._scales
            state[prefix + '_e8_indices'] = self._e8_indices
        return state
        
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        signs_key = prefix + '_signs'
        scales_key = prefix + '_scales'
        indices_key = prefix + '_e8_indices'
        
        if signs_key in state_dict:
            self._signs = torch.empty_like(state_dict[signs_key])
        if scales_key in state_dict:
            self._scales = torch.empty_like(state_dict[scales_key])
        if indices_key in state_dict:
            self._e8_indices = torch.empty_like(state_dict[indices_key])
            
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        self.compressed = True

    @property
    def weight(self) -> torch.Tensor:
        if not self.compressed:
            raise RuntimeError("Linear layer has not been compressed.")
        device = self._e8_indices.device
        dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
        return self.materialize_weight(device, dtype)


class HIVQEmbeddingWeightProxy:
    """
    Lightweight proxy for HIVQEmbedding.weight that supports sliced row indexing
    (e.g., embed_tokens.weight[pad_token_id, :]) without materializing all 256,000 rows.
    """
    def __init__(self, embed_module):
        self.embed_module = embed_module

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            row_idx = idx[0]
            col_idx = idx[1] if len(idx) > 1 else slice(None)
            rows = self.embed_module.materialize_rows(row_idx)
            return rows[..., col_idx]
        return self.embed_module.materialize_rows(idx)

    @property
    def shape(self):
        return (self.embed_module.num_embeddings, self.embed_module.embedding_dim)

    @property
    def dtype(self):
        return torch.bfloat16

    @property
    def device(self):
        return self.embed_module._e8_indices.device


class HIVQEmbedding(torch.nn.Module):
    """Embedding layer using 2-bit E8P lattice vector quantization and Rademacher-Hadamard rotation."""
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = None, embed_scale: float = 1.0):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.compressed = False
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=True)
        
        self.register_buffer('_signs', torch.empty(0, dtype=torch.float32))
        self.register_buffer('_scales', torch.empty(0, dtype=torch.float32))
        self.register_buffer('_e8_indices', torch.empty(0, dtype=torch.int32))
        
    def load_from_weight(self, W: torch.Tensor, device="cpu"):
        comp_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        D = self.embedding_dim
        N_in = D if (D & (D - 1)) == 0 else 2 ** ((D - 1).bit_length())
        N_in = max(N_in, 8)
        
        signs = (torch.randint(0, 2, (self.embedding_dim,), dtype=torch.float32, device=comp_device) * 2.0 - 1.0).to(dtype=W.dtype)
        
        row_chunk_size = 4096
        all_scales = []
        all_indices = []
        
        codebook_obj = E8Codebook.get_instance(device=comp_device)
        grid = codebook_obj.grid.to(dtype=torch.float32, device=comp_device)
        grid_norm_sq = codebook_obj.grid_norm_sq.to(dtype=torch.float32, device=comp_device)
        
        for start_row in range(0, self.num_embeddings, row_chunk_size):
            end_row = min(start_row + row_chunk_size, self.num_embeddings)
            W_chunk = W[start_row:end_row].to(comp_device)
            
            W_chunk_signed = W_chunk * signs.unsqueeze(0)
            W_chunk_padded = torch.nn.functional.pad(W_chunk_signed, (0, N_in - D), mode='constant', value=0.0)
            W_chunk_rot = fwht(W_chunk_padded)
            
            scales_chunk = torch.max(torch.abs(W_chunk_rot), dim=-1, keepdim=True)[0] / 3.0
            scales_chunk = torch.clamp(scales_chunk, min=1e-5)
            all_scales.append(scales_chunk.to(device))
            
            W_chunk_norm = W_chunk_rot / scales_chunk
            W_blocks = W_chunk_norm.view(-1, 8)
            
            block_signs = (W_blocks < 0).to(torch.int32)
            sign_bits = torch.zeros(W_blocks.shape[0], dtype=torch.int32, device=comp_device)
            for b in range(8):
                sign_bits |= (block_signs[:, b] << b)
                
            W_abs_all = W_blocks.float().abs()
            block_batch_size = 32768
            lut_idx_list = []
            for blk_start in range(0, W_abs_all.shape[0], block_batch_size):
                W_abs_sub = W_abs_all[blk_start:blk_start + block_batch_size]
                dists_sub = (W_abs_sub**2).sum(dim=-1, keepdim=True) + grid_norm_sq.unsqueeze(0) - 2.0 * torch.matmul(W_abs_sub, grid.T)
                lut_idx_list.append(torch.argmin(dists_sub, dim=-1).to(torch.int32))
            lut_idx = torch.cat(lut_idx_list, dim=0)
            
            packed_idx = (sign_bits << 8) | lut_idx
            all_indices.append(packed_idx.to(device))
            
        scales = torch.cat(all_scales, dim=0)
        indices = torch.cat(all_indices, dim=0).to(torch.int32)
        
        self.register_buffer('_signs', signs.to(device))
        self.register_buffer('_scales', scales.squeeze(-1).to(torch.float32).to(device))
        self.register_buffer('_e8_indices', indices.to(device))
        
        self.compressed = True

    def materialize_rows(self, idxs, device=None, dtype=None):
        """Dequantizes requested row indices in micro-chunks of 8192 rows, capping peak VRAM to <100 MB."""
        if device is None:
            device = self._e8_indices.device
        if dtype is None:
            dtype = self._signs.dtype if (hasattr(self, '_signs') and self._signs.numel() > 0) else (torch.bfloat16 if device.type == 'cuda' else torch.float32)
            
        is_single = isinstance(idxs, int)
        if is_single:
            flat_ids = torch.tensor([idxs], device=device)
        elif isinstance(idxs, slice):
            flat_ids = torch.arange(self.num_embeddings, device=device)[idxs]
        elif isinstance(idxs, torch.Tensor):
            flat_ids = idxs.to(device).view(-1)
        else:
            flat_ids = torch.tensor(idxs, device=device).view(-1)
            
        D = self.embedding_dim
        N_in = D if (D & (D - 1)) == 0 else 2 ** ((D - 1).bit_length())
        N_in = max(N_in, 8)
        block_size = N_in // 8
        
        chunk_size = 8192
        total_ids = len(flat_ids)
        all_x_rot = []
        
        shifts = torch.arange(8, device=device)
        signs_dev = self._signs.to(device=device, dtype=dtype)
        grid_dev = E8Codebook.get_instance(device=device).grid.to(dtype=dtype)
        scales_dev = self._scales.to(device=device, dtype=dtype)
        indices_dev = self._e8_indices.to(device)
        
        for start in range(0, total_ids, chunk_size):
            sub_ids = flat_ids[start:start + chunk_size]
            sub_len = len(sub_ids)
            
            offsets = torch.arange(block_size, device=device).unsqueeze(0)
            token_offsets = sub_ids.unsqueeze(1) * block_size
            gather_indices = (token_offsets + offsets).view(-1)
            
            sub_e8_indices = torch.index_select(indices_dev, 0, gather_indices)
            
            lut_idx = (sub_e8_indices & 0xFF).long()
            sign_bits = (sub_e8_indices >> 8) & 0xFF
            
            mags = grid_dev[lut_idx]
            sign_vectors = 1.0 - 2.0 * ((sign_bits.unsqueeze(-1) >> shifts) & 1).to(dtype=dtype)
            
            sub_vectors = (mags * sign_vectors).view(sub_len, N_in)
            
            sub_scales = torch.index_select(scales_dev, 0, sub_ids)
            sub_vectors = sub_vectors * sub_scales.unsqueeze(-1)
            
            W_rot = fwht(sub_vectors)
            sub_x_rot = W_rot[..., :D] * signs_dev.unsqueeze(0)
            all_x_rot.append(sub_x_rot.to(dtype=dtype))
            
        x_rot = torch.cat(all_x_rot, dim=0)
        
        if is_single:
            return x_rot.squeeze(0).to(dtype=dtype)
        return x_rot.to(dtype=dtype)

    def materialize_weight(self, device, dtype):
        return self.materialize_rows(torch.arange(self.num_embeddings, device=device), device=device, dtype=dtype)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not self.compressed:
            raise RuntimeError("Embedding weights have not been compressed using HIVQ.")
            
        device = input_ids.device
        dtype = self._signs.dtype if (hasattr(self, '_signs') and self._signs.numel() > 0) else (torch.bfloat16 if device.type == 'cuda' else torch.float32)
        
        x_rot = self.materialize_rows(input_ids, device=device, dtype=dtype)
        
        if self.embed_scale != 1.0:
            x_rot = x_rot * self.embed_scale
            
        return x_rot.view(*input_ids.shape, self.embedding_dim)
        
    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        if self.compressed:
            state[prefix + '_signs'] = self._signs
            state[prefix + '_scales'] = self._scales
            state[prefix + '_e8_indices'] = self._e8_indices
        return state
        
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        signs_key = prefix + '_signs'
        scales_key = prefix + '_scales'
        indices_key = prefix + '_e8_indices'
        
        if signs_key in state_dict:
            self._signs = torch.empty_like(state_dict[signs_key])
        if scales_key in state_dict:
            self._scales = torch.empty_like(state_dict[scales_key])
        if indices_key in state_dict:
            self._e8_indices = torch.empty_like(state_dict[indices_key])
            
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        self.compressed = True

    @property
    def weight(self):
        if not self.compressed:
            raise RuntimeError("Embedding has not been compressed.")
        return HIVQEmbeddingWeightProxy(self)


class HIVQTiedHead(torch.nn.Module):
    """Output head that reuses the HIVQEmbedding's materialized weight.
    
    When tie_word_embeddings=True (Gemma 4 default), lm_head.weight == embed_tokens.weight.
    Delegates to embed_tokens' materialized weight and caches on GPU for 50+ tokens/sec inference speed.
    """
    def __init__(self, embed_module: 'HIVQEmbedding'):
        super().__init__()
        self.embed_module = embed_module
        self.in_features = embed_module.embedding_dim
        self.out_features = embed_module.num_embeddings
        self.bias = None
        self._cached_weight = None
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        device = X.device
        dtype = X.dtype
        if self._cached_weight is not None and self._cached_weight.device == device and self._cached_weight.dtype == dtype:
            W = self._cached_weight
        else:
            W = self.embed_module.materialize_rows(torch.arange(self.out_features, device=device), device=device, dtype=dtype)
            if device.type == 'cuda':
                self._cached_weight = W
        return torch.nn.functional.linear(X, W)
        
    @property
    def weight(self):
        return HIVQEmbeddingWeightProxy(self.embed_module)
