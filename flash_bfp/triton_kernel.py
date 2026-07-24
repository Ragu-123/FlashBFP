import torch
from typing import Dict

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    # Mock triton modules for CPU/Windows import compatibility
    class DummyModule:
        def __getattr__(self, name):
            if name == 'constexpr':
                return int
            return lambda *args, **kwargs: None
            
    tl = DummyModule()
    triton = DummyModule()
    # No-op decorator
    triton.jit = lambda x: x

@triton.jit
def oabf_gemm_kernel(
    x_ptr, y_ptr,
    exponents_ptr, signs_ptr, payload_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Triton kernel that performs GEMM directly on OABF compressed weights.
    Decompression occurs entirely inside GPU registers/SRAM.
    All pointer arguments must be float32/int32 CUDA tensors.
    """
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # --- Load activation tile with boundary mask ---
        a_offs_k = k * BLOCK_SIZE_K + offs_k
        a_ptr = x_ptr + offs_am[:, None] * stride_am + a_offs_k[None, :] * stride_ak
        a_mask = (offs_am[:, None] < M) & (a_offs_k[None, :] < K)
        a_tile = tl.load(a_ptr, mask=a_mask, other=0.0)
        
        w_tile = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
        
        # Total number of blocks in the compressed payload
        num_k_blocks = K // 16
        
        for step in range(0, BLOCK_SIZE_K // 16):
            k_block = k * (BLOCK_SIZE_K // 16) + step
            block_idx = offs_bn[None, :] * num_k_blocks + k_block
            
            # Bounds mask: prevent reading past the payload buffer
            # Max valid block_idx = (C_padded - 1) * num_k_blocks + (num_k_blocks - 1)
            # = C_padded * num_k_blocks - 1 = total_blocks - 1
            total_blocks = N * num_k_blocks
            b_mask = (offs_bn[None, :] < N) & (k_block < num_k_blocks)
            
            exp = tl.load(exponents_ptr + block_idx, mask=b_mask, other=0.0).to(tl.float32)
            signs = tl.load(signs_ptr + block_idx, mask=b_mask, other=0).to(tl.int32)
            
            p_mask = b_mask  # Same mask for payload (2 words per block)
            p_word1 = tl.load(payload_ptr + block_idx * 2, mask=p_mask, other=0)
            p_word2 = tl.load(payload_ptr + block_idx * 2 + 1, mask=p_mask, other=0)
            
            for i in range(8):
                mantissa = (p_word1 >> (i * 4)) & 0xF
                sign = (signs >> i) & 1
                val = tl.where(sign == 1, -1.0, 1.0) * (mantissa.to(tl.float32) / 15.0) * tl.exp2(exp)
                
                row_idx = step * 16 + i
                w_tile = tl.where(tl.arange(0, BLOCK_SIZE_K)[:, None] == row_idx, val, w_tile)
                
            for i in range(8):
                mantissa = (p_word2 >> (i * 4)) & 0xF
                sign = (signs >> (i + 8)) & 1
                val = tl.where(sign == 1, -1.0, 1.0) * (mantissa.to(tl.float32) / 15.0) * tl.exp2(exp)
                
                row_idx = step * 16 + i + 8
                w_tile = tl.where(tl.arange(0, BLOCK_SIZE_K)[:, None] == row_idx, val, w_tile)
                
        accumulator += tl.dot(a_tile, w_tile.to(a_tile.dtype))
        
    c_ptr = y_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptr, accumulator, mask=c_mask)


def _cpu_fallback_gemm(X: torch.Tensor, dense_payload: Dict) -> torch.Tensor:
    """
    Pure-PyTorch CPU/CUDA fallback: decompress weights and run standard matmul.
    Used when Triton is unavailable or when tensors cannot be placed on CUDA.
    """
    R_padded, C_padded = dense_payload['padded_shape']
    C_orig, R_orig = dense_payload['orig_shape']
    
    exponents = dense_payload['exponents'].to(device=X.device, dtype=torch.float32)
    signs_packed = dense_payload['signs'].to(device=X.device, dtype=torch.int32)
    payload = dense_payload['payload'].to(device=X.device, dtype=torch.int32)
    
    num_blocks = payload.shape[0] // 2 if payload.dim() == 1 else payload.shape[0]
    
    unpacked_blocks = torch.zeros((num_blocks, 16), dtype=X.dtype, device=X.device)
    
    if payload.dim() == 1:
        p0 = payload[0::2]  # even indices = word1
        p1 = payload[1::2]  # odd indices = word2
    else:
        p0 = payload[:, 0]
        p1 = payload[:, 1]
    
    for i in range(8):
        mantissa = (p0 >> (i * 4)) & 0xF
        sign = (signs_packed >> i) & 1
        val = torch.where(sign == 1, -1.0, 1.0) * (mantissa.to(X.dtype) / 15.0) * torch.pow(2.0, exponents)
        unpacked_blocks[:, i] = val
        
    for i in range(8):
        mantissa = (p1 >> (i * 4)) & 0xF
        sign = (signs_packed >> (i + 8)) & 1
        val = torch.where(sign == 1, -1.0, 1.0) * (mantissa.to(X.dtype) / 15.0) * torch.pow(2.0, exponents)
        unpacked_blocks[:, i + 8] = val
        
    W_reconstructed = unpacked_blocks.flatten().view(C_padded, R_padded)
    W_reconstructed = W_reconstructed[:C_orig, :R_orig]
    
    return torch.matmul(X, W_reconstructed.t())


def oabf_gemm(X: torch.Tensor, dense_payload: Dict) -> torch.Tensor:
    """
    PyTorch wrapper that configures and launches the Triton fused OABF GEMM kernel.
    Falls back to pure-PyTorch matmul when Triton is unavailable or tensors are on CPU.
    All tensors are explicitly synced to X's device and made contiguous before launch.
    """
    # --- Decide execution path ---
    # Use Triton GPU path ONLY when X is on CUDA, Triton is available, AND
    # we can successfully place all payload tensors on the same CUDA device.
    use_triton = HAS_TRITON and X.is_cuda
    
    if not use_triton:
        return _cpu_fallback_gemm(X, dense_payload)
    
    target_device = X.device
    
    # --- Explicit device sync + contiguous for ALL tensors ---
    try:
        exponents = dense_payload['exponents'].to(device=target_device, dtype=torch.float32).contiguous()
        signs = dense_payload['signs'].to(device=target_device, dtype=torch.int32).contiguous()
        payload_raw = dense_payload['payload'].to(device=target_device, dtype=torch.int32).contiguous()
        # Flatten payload to 1D for direct pointer arithmetic in the kernel
        payload_flat = payload_raw.flatten().contiguous()
    except RuntimeError:
        # If .to(cuda) fails for any reason (unsupported dtype, OOM, etc.), use CPU fallback
        return _cpu_fallback_gemm(X, dense_payload)
    
    # Verify ALL tensors are actually on CUDA before launching kernel
    if not (exponents.is_cuda and signs.is_cuda and payload_flat.is_cuda):
        return _cpu_fallback_gemm(X, dense_payload)
    
    X = X.contiguous()
    
    M, K_x = X.shape
    R_padded, C_padded = dense_payload['padded_shape']
    C_orig, R_orig = dense_payload['orig_shape']
    
    if K_x < R_padded:
        X = torch.nn.functional.pad(X, (0, R_padded - K_x), mode='constant', value=0.0)
        
    Y = torch.empty((M, C_padded), device=target_device, dtype=X.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(C_padded, META['BLOCK_SIZE_N']),)
    
    oabf_gemm_kernel[grid](
        X, Y,
        exponents,
        signs,
        payload_flat,
        M, C_padded, R_padded,
        X.stride(0), X.stride(1),
        Y.stride(0), Y.stride(1),
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=16
    )
    
    return Y[:, :C_orig]


class OABFLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        
        self.compressed = False
        self._payload_meta = {}
        self._sparse_meta = {}
        
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def load_from_weight(self, W: torch.Tensor, compressor, device):
        """
        Compresses and loads weight matrix W into this linear layer on-the-fly.
        Compression runs entirely on the CPU to prevent device mismatch errors.
        
        All tensors are CAST TO STANDARD DTYPES (float32, int32) before moving
        to the target device. This guarantees CUDA compatibility across all
        PyTorch versions and prevents Triton pointer access errors from exotic
        dtypes like int8, int16, uint16 or uint32.
        """
        # Meta tensors have shape but no data — materialize as zeros on CPU
        if W.device.type == 'meta':
            W = torch.zeros(W.shape, dtype=W.dtype, device='cpu')
        dense_payload, sparse_outliers = compressor.compress_matrix(W.cpu())
        
        # Cast to universally-supported dtypes BEFORE .to(device)
        # int8 exponents -> float32 (Triton loads as float32 anyway)
        self.register_buffer('_dp_exponents', dense_payload['exponents'].to(torch.float32).to(device))
        
        # Store integer tensors as standard attributes rather than registered buffers
        # to prevent Hugging Face Accelerate's AlignDevicesHook from casting them to float16/bfloat16.
        self._dp_signs = dense_payload['signs'].to(torch.int32).to(device)
        self._dp_payload = dense_payload['payload'].to(torch.int32).flatten().to(device)
        
        # Store shape metadata (non-tensor, not affected by device moves)
        self._payload_meta = {
            'padded_shape': dense_payload['padded_shape'],
            'orig_shape': dense_payload['orig_shape'],
        }
        
        # Sparse outlier tensors — also stored as standard attributes
        self._sp_block_idx = sparse_outliers['block_idx'].to(torch.int32).to(device)
        self._sp_offset = sparse_outliers['offset'].to(torch.int32).to(device)
        self.register_buffer('_sp_value', sparse_outliers['value'].to(torch.float32).to(device))
        self._sparse_meta = {
            'count': sparse_outliers['count'],
        }
        
        # Strict CPU-side validation of outlier indices at load time to prevent any CUDA asserts
        if sparse_outliers['count'] > 0:
            block_idx_cpu = sparse_outliers['block_idx'].long()
            offset_cpu = sparse_outliers['offset'].long()
            padded_R, padded_C = dense_payload['padded_shape']
            C_orig, R_orig = dense_payload['orig_shape']
            
            global_idx_cpu = block_idx_cpu * 16 + offset_cpu
            row_idx_cpu = global_idx_cpu % padded_R
            col_idx_cpu = global_idx_cpu // padded_R
            
            # Check bounds against padded shape
            if (row_idx_cpu < 0).any() or (row_idx_cpu >= padded_R).any():
                raise IndexError(f"FlashBFP Compressor Error: Outlier row index out of padded bounds [0, {padded_R-1}]")
            if (col_idx_cpu < 0).any() or (col_idx_cpu >= padded_C).any():
                raise IndexError(f"FlashBFP Compressor Error: Outlier column index out of padded bounds [0, {padded_C-1}]")
        
        self.compressed = True

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Executes fused OABF Linear forward pass:
        1. Sync all tensors to a common execution device.
        2. Run Fused Triton GEMM (dense path) or CPU fallback.
        3. Run Sparse Outliers SpMV (sparse path).
        4. Add Bias.
        """
        if not self.compressed:
            raise RuntimeError("Weight matrix has not been compressed and loaded.")
            
        orig_shape = X.shape
        input_device = X.device
        
        # Determine execution device: use our buffer's device as truth.
        # Accelerate's hooks move registered buffers, so trust their location.
        exec_device = self._dp_exponents.device
        
        # Explicitly move non-buffer attributes to exec_device if needed.
        # Since they are not registered buffers, Accelerate will not touch them.
        if self._dp_signs.device != exec_device:
            self._dp_signs = self._dp_signs.to(exec_device)
        if self._dp_payload.device != exec_device:
            self._dp_payload = self._dp_payload.to(exec_device)
            
        X_flat = X.view(-1, self.in_features)
        
        # Move X to the execution device if accelerate placed them differently
        if X_flat.device != exec_device:
            X_flat = X_flat.to(exec_device)
        
        live_payload = {
            'exponents': self._dp_exponents,
            'signs': self._dp_signs,
            'payload': self._dp_payload,
            'padded_shape': self._payload_meta['padded_shape'],
            'orig_shape': self._payload_meta['orig_shape'],
        }
        Y_dense = oabf_gemm(X_flat, live_payload)
        
        if self._sparse_meta['count'] > 0:
            R_padded, C_padded = self._payload_meta['padded_shape']
            
            # Move outlier attributes to exec_device and cache the location
            if self._sp_block_idx.device != exec_device:
                self._sp_block_idx = self._sp_block_idx.to(exec_device)
            if self._sp_offset.device != exec_device:
                self._sp_offset = self._sp_offset.to(exec_device)
                
            outlier_block = self._sp_block_idx
            outlier_offset = self._sp_offset
            outlier_val = self._sp_value
            
            if outlier_val.device != exec_device:
                outlier_val = outlier_val.to(exec_device)
            
            # Use int64 for ALL index arithmetic to prevent overflow
            # on large layers (MLP gate/up/down have 82M+ elements)
            global_indices = outlier_block.long() * 16 + outlier_offset.long()
            row_indices = global_indices % R_padded
            col_indices = global_indices // R_padded
            
            # Strict boundary validation including >= 0 bounds checking
            valid_mask = (row_indices >= 0) & (row_indices < self.in_features) & \
                         (col_indices >= 0) & (col_indices < self.out_features)
            
            # Ensure integer attributes remain correct type (int64 for indexing)
            r_idx = row_indices[valid_mask].long()
            c_idx = col_indices[valid_mask].long()
            v_val = outlier_val[valid_mask].to(X_flat.dtype)
            
            # Ironclad safety clamp to prevent CUDA out-of-bounds indexing assertions
            if c_idx.numel() > 0:
                c_idx = torch.clamp(c_idx, 0, self.out_features - 1)
                r_idx = torch.clamp(r_idx, 0, self.in_features - 1)
            
            X_active = X_flat[:, r_idx]
            products = X_active * v_val[None, :]
            
            Y_sparse = torch.zeros_like(Y_dense)
            if c_idx.numel() > 0:
                Y_sparse.index_add_(1, c_idx, products)
            
            Y_dense = Y_dense + Y_sparse
            
        if self.bias is not None:
            bias = self.bias
            if bias.device != exec_device:
                bias = bias.to(exec_device)
            Y_dense = Y_dense + bias
        
        # Move output back to input device if they differ (for accelerate hooks)
        if Y_dense.device != input_device:
            Y_dense = Y_dense.to(input_device)
            
        return Y_dense.view(*orig_shape[:-1], self.out_features)
