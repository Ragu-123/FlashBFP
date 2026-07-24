import torch
import numpy as np

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

def pad_ifwht(y):
    """Inverse of pad_fwht using normalized Walsh-Hadamard symmetry."""
    D = y.shape[-1]
    if (D & (D - 1)) == 0:
        return fwht(y)
    N = 2 ** ((D - 1).bit_length())
    y_padded = torch.nn.functional.pad(y, (0, N - D), mode='constant', value=0.0)
    y_rot = fwht(y_padded)
    return y_rot[..., :D]

def decode_d8_pytorch(v):
    """Decodes 8D vector v to the closest point in the D8 checkerboard lattice (sum must be even)."""
    rounded = torch.round(v)
    parity_sum = torch.sum(rounded, dim=-1)
    
    odd_mask = (torch.remainder(parity_sum.long(), 2) != 0)
    
    if odd_mask.any():
        errors = torch.abs(v - rounded)
        worst_idx = torch.argmax(errors, dim=-1)
        
        adjust = torch.where(rounded > v, -1.0, 1.0)
        one_hot = torch.nn.functional.one_hot(worst_idx, num_classes=8).to(v.device).to(v.dtype)
        rounded = rounded + adjust * one_hot * odd_mask.unsqueeze(-1).to(v.dtype)
        
    return rounded

def conway_sloane_e8(x):
    """Fast Conway-Sloane decoder for E8 lattice (union of D8 and D8 + 0.5)."""
    c0 = decode_d8_pytorch(x)
    c1 = decode_d8_pytorch(x - 0.5) + 0.5
    
    dist0 = torch.sum((x - c0)**2, dim=-1)
    dist1 = torch.sum((x - c1)**2, dim=-1)
    
    use_c0 = (dist0 <= dist1).unsqueeze(-1)
    return torch.where(use_c0, c0, c1)


class E8Codebook:
    """Generates and manages the 16-bit (2-bit per element) E8 lattice codebook mapping."""
    _instance = None
    
    @classmethod
    def get_instance(cls, device="cpu"):
        if cls._instance is None:
            cls._instance = cls(device=device)
        else:
            cls._instance.to(device)
        return cls._instance
        
    def __init__(self, device="cpu"):
        # Generate the first few shells of E8 to populate a codebook of size 65536
        points = []
        
        # Shell 0: Origin (1 point)
        points.append(np.zeros(8))
        
        # Shell 2: 240 points
        # 1. Permutations of (+-1, +-1, 0, 0, 0, 0, 0, 0) -> 112 points
        for i in range(8):
            for j in range(i + 1, 8):
                for s1 in [-1.0, 1.0]:
                    for s2 in [-1.0, 1.0]:
                        p = np.zeros(8)
                        p[i] = s1
                        p[j] = s2
                        points.append(p)
                        
        # 2. Permutations of (+-0.5, ..., +-0.5) with even parity -> 128 points
        for b in range(256):
            # Generate all 8-bit signs
            signs = np.array([1.0 if (b & (1 << i)) else -1.0 for i in range(8)])
            if np.sum(signs > 0) % 2 == 0:  # Even number of positive signs
                points.append(signs * 0.5)
                
        # Fill remaining slots with Shell 4 points until we reach 65536
        # Shell 4: 2160 points
        # Permutations of (+-2, 0, 0, 0, 0, 0, 0, 0) -> 16 points
        for i in range(8):
            for s in [-2.0, 2.0]:
                p = np.zeros(8)
                p[i] = s
                points.append(p)
                
        # Permutations of (+-1, +-1, +-1, +-1, 0, 0, 0, 0) -> 1120 points
        from itertools import combinations
        for idxs in combinations(range(8), 4):
            for s_mask in range(16):
                p = np.zeros(8)
                for bit in range(4):
                    p[idxs[bit]] = 1.0 if (s_mask & (1 << bit)) else -1.0
                points.append(p)
                
        # Pad up to exactly 65536 points with zeros
        while len(points) < 65536:
            points.append(np.zeros(8))
            
        # Store as a tensor
        self.codebook = torch.tensor(np.array(points[:65536]), dtype=torch.float32, device=device)
        
    def to(self, device):
        self.codebook = self.codebook.to(device)
        return self


class HIVQLinear(torch.nn.Module):
    """Linear layer using 2-bit E8 lattice vector quantization and Rademacher-Hadamard rotation."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        self.compressed = False
        
        # Register empty buffers for state dict loading/resizing
        self.register_buffer('_signs', torch.empty(0, dtype=torch.float32))
        self.register_buffer('_scales', torch.empty(0, dtype=torch.float32))
        self.register_buffer('_e8_indices', torch.empty(0, dtype=torch.int32))
        
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
            
    def load_from_weight(self, W: torch.Tensor, device="cpu"):
        """Rotates weight using RHT, maps to E8 lattice, and saves compressed representations."""
        comp_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 1. Generate Rademacher signs on GPU
        signs = torch.randint(0, 2, (self.in_features,), dtype=torch.float32, device=comp_device) * 2.0 - 1.0
        
        # 2. Apply Rademacher signs and forward FWHT on GPU
        W_signed = W.to(comp_device) * signs.unsqueeze(0)
        W_rot = pad_fwht(W_signed)
        
        # 3. Calculate row-wise scale factors on GPU
        scales = torch.max(torch.abs(W_rot), dim=-1, keepdim=True)[0] / 3.0
        scales = torch.clamp(scales, min=1e-5)
        
        W_norm = W_rot / scales
        
        # 4. Map 8D blocks to E8 on GPU
        W_norm_grouped = W_norm.view(-1, 8)
        e8_points = conway_sloane_e8(W_norm_grouped)
        
        # 5. Map E8 points to closest codebook index on GPU
        # Run search on GPU in float16 to leverage Tensor Cores and reduce memory overhead
        comp_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        codebook = E8Codebook.get_instance(device=comp_device).codebook
        
        # Cast to float16 for ultra-fast GPU search
        codebook_half = codebook.to(dtype=torch.float16, device=comp_device)
        e8_points_half = e8_points.to(dtype=torch.float16, device=comp_device)
        codebook_norms = torch.sum(codebook_half**2, dim=-1) # [65536]
        
        indices = []
        batch_size = 16384  # Large batch size to maximize GPU execution and minimize loop overhead
        for i in range(0, e8_points_half.shape[0], batch_size):
            block = e8_points_half[i:i+batch_size] # [B, 8]
            block_norms = torch.sum(block**2, dim=-1, keepdim=True) # [B, 1]
            
            # Distance expansion in float16: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x . y
            dists = block_norms + codebook_norms.unsqueeze(0) - 2.0 * torch.matmul(block, codebook_half.T) # [B, 65536]
            indices.append(torch.argmin(dists, dim=-1))
            
        indices = torch.cat(indices, dim=0).to(torch.int32).to(device)
        
        # Move final buffers to target device (Accelereate managed sharded device)
        self.register_buffer('_signs', signs.to(device))
        self.register_buffer('_scales', scales.squeeze(-1).to(torch.float32).to(device))
        self.register_buffer('_e8_indices', indices.to(device))
        
        self.compressed = True
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if not self.compressed:
            raise RuntimeError("Layer weights have not been compressed using HIVQ.")
            
        orig_shape = X.shape
        X = X.view(-1, self.in_features)
        
        # Cast/move buffers to match input device and dtype (float32/bfloat16) on-the-fly to support device_map='auto'
        device = X.device
        dtype = X.dtype
        
        signs = self._signs.to(device=device, dtype=dtype)
        scales = self._scales.to(device=device, dtype=dtype)
        e8_indices = self._e8_indices.to(device=device)
        
        # 1. Apply Rademacher-Hadamard rotation to activation inputs
        X_signed = X * signs.unsqueeze(0)
        X_rot = pad_fwht(X_signed)
        
        # 2. Dequantize weights from 16-bit indices on-the-fly using E8 codebook
        codebook = E8Codebook.get_instance(device=device).codebook.to(dtype=dtype)
        W_dequant = codebook[e8_indices.long()] # [out_features * in_features // 8, 8]
        W_dequant = W_dequant.view(self.out_features, self.in_features)
        
        # Scale dequantized weights row-wise
        W_dequant = W_dequant * scales.unsqueeze(-1)
        
        # 3. Perform rotated GEMM
        Y_rot = torch.nn.functional.linear(X_rot, W_dequant)
        
        # 4. Add bias
        if self.bias is not None:
            bias_cast = self.bias.to(device=device, dtype=dtype)
            Y_rot = Y_rot + bias_cast
            
        return Y_rot.view(*orig_shape[:-1], self.out_features)
        
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


class HIVQEmbedding(torch.nn.Module):
    """Embedding layer using 2-bit E8 lattice vector quantization and Rademacher-Hadamard rotation."""
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.compressed = False
        
        # Register empty buffers for state dict loading/resizing
        self.register_buffer('_signs', torch.empty(0, dtype=torch.float32))
        self.register_buffer('_scales', torch.empty(0, dtype=torch.float32))
        self.register_buffer('_e8_indices', torch.empty(0, dtype=torch.int32))
        
    def load_from_weight(self, W: torch.Tensor, device="cpu"):
        """Rotates embedding weights offline using RHT and maps to E8 lattice representation."""
        comp_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 1. Generate Rademacher signs on GPU
        signs = torch.randint(0, 2, (self.embedding_dim,), dtype=torch.float32, device=comp_device) * 2.0 - 1.0
        
        # 2. Apply Rademacher signs and forward FWHT on GPU
        W_signed = W.to(comp_device) * signs.unsqueeze(0)
        W_rot = pad_fwht(W_signed)
        
        # 3. Calculate row-wise scale factors on GPU
        scales = torch.max(torch.abs(W_rot), dim=-1, keepdim=True)[0] / 3.0
        scales = torch.clamp(scales, min=1e-5)
        
        W_norm = W_rot / scales
        
        # 4. Map 8D blocks to E8 on GPU
        W_norm_grouped = W_norm.view(-1, 8)
        e8_points = conway_sloane_e8(W_norm_grouped)
        
        # 5. Map E8 points to closest codebook index on GPU
        codebook = E8Codebook.get_instance(device=comp_device).codebook
        codebook_norms = torch.sum(codebook**2, dim=-1) # [65536]
        
        indices = []
        batch_size = 16384  # Keep batch size large to leverage GPU parallelization
        for i in range(0, e8_points.shape[0], batch_size):
            block = e8_points[i:i+batch_size] # [B, 8]
            block_norms = torch.sum(block**2, dim=-1, keepdim=True) # [B, 1]
            
            dists = block_norms + codebook_norms.unsqueeze(0) - 2.0 * torch.matmul(block, codebook.T) # [B, 65536]
            indices.append(torch.argmin(dists, dim=-1))
            
        indices = torch.cat(indices, dim=0).to(torch.int32).to(device)
        
        # Move final buffers to target device
        self.register_buffer('_signs', signs.to(device))
        self.register_buffer('_scales', scales.squeeze(-1).to(torch.float32).to(device))
        self.register_buffer('_e8_indices', indices.to(device))
        
        self.compressed = True
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not self.compressed:
            raise RuntimeError("Embedding weights have not been compressed using HIVQ.")
            
        device = input_ids.device
        dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
        
        signs = self._signs.to(device=device, dtype=dtype)
        scales = self._scales.to(device=device, dtype=dtype)
        e8_indices = self._e8_indices.to(device=device)
        
        block_size = self.embedding_dim // 8
        flat_ids = input_ids.view(-1)
        
        # Vectorized gather of contiguous E8 indices for target token IDs
        offsets = torch.arange(block_size, device=device).unsqueeze(0) # [1, block_size]
        token_offsets = flat_ids.unsqueeze(1) * block_size # [num_tokens, 1]
        gather_indices = (token_offsets + offsets).view(-1) # [num_tokens * block_size]
        
        token_e8_indices = torch.index_select(e8_indices, 0, gather_indices) # [num_tokens * block_size]
        
        # Dequantize E8 points using E8 codebook lookup
        codebook = E8Codebook.get_instance(device=device).codebook.to(dtype=dtype)
        token_vectors = codebook[token_e8_indices.long()] # [num_tokens * block_size, 8]
        token_vectors = token_vectors.view(-1, self.embedding_dim) # [num_tokens, embedding_dim]
        
        # Scale dequantized vectors row-wise
        token_scales = torch.index_select(scales, 0, flat_ids) # [num_tokens]
        token_vectors = token_vectors * token_scales.unsqueeze(-1)
        
        # 4. Apply inverse Rademacher-Hadamard rotation online in GPU SRAM
        x_rot = pad_ifwht(token_vectors) * signs.unsqueeze(0)
        
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
