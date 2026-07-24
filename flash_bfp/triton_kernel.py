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
        
        # Total number of blocks in the compressed payload
        num_k_blocks = K // 16
        
        w_tile = None
        
        for step in range(0, BLOCK_SIZE_K // 16):
            k_block = k * (BLOCK_SIZE_K // 16) + step
            block_idx = offs_bn[None, :] * num_k_blocks + k_block
            
            # Bounds mask: prevent reading past the payload buffer
            b_mask = (offs_bn[None, :] < N) & (k_block < num_k_blocks)
            
            exp = tl.load(exponents_ptr + block_idx, mask=b_mask, other=0.0).to(tl.float32)
            signs = tl.load(signs_ptr + block_idx, mask=b_mask, other=0).to(tl.int32)
            
            p_mask = b_mask  # Same mask for payload (2 words per block)
            p_word1 = tl.load(payload_ptr + block_idx * 2, mask=p_mask, other=0)
            p_word2 = tl.load(payload_ptr + block_idx * 2 + 1, mask=p_mask, other=0)
            
            # Vectorized bitwise unpacking of 16 elements per block
            shift_matrix = tl.arange(0, 8)[:, None] * 4
            i_arr = tl.arange(0, 8)[:, None]
            
            # Unpack first 8 mantissas and signs
            mantissas1 = (p_word1 >> shift_matrix) & 0xF
            signs1 = (signs >> i_arr) & 1
            val1 = tl.where(signs1 == 1, -1.0, 1.0) * (mantissas1.to(tl.float32) / 15.0) * tl.exp2(exp)
            
            # Unpack second 8 mantissas and signs
            mantissas2 = (p_word2 >> shift_matrix) & 0xF
            signs2 = (signs >> (i_arr + 8)) & 1
            val2 = tl.where(signs2 == 1, -1.0, 1.0) * (mantissas2.to(tl.float32) / 15.0) * tl.exp2(exp)
            
            # Concatenate along the row dimension to get a (16, BLOCK_SIZE_N) block
            val_block = tl.cat(val1, val2, dim=0)
            
            if w_tile is None:
                w_tile = val_block
            else:
                w_tile = tl.cat(w_tile, val_block, dim=0)
                
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
    
    # --- Fast-path device/contiguous checks to avoid redundant allocations ---
    try:
        exponents = dense_payload['exponents']
        if exponents.device != target_device or exponents.dtype != torch.float32 or not exponents.is_contiguous():
            exponents = exponents.to(device=target_device, dtype=torch.float32).contiguous()
            
        signs = dense_payload['signs']
        if signs.device != target_device or signs.dtype != torch.int32 or not signs.is_contiguous():
            signs = signs.to(device=target_device, dtype=torch.int32).contiguous()
            
        payload = dense_payload['payload']
        if payload.device != target_device or payload.dtype != torch.int32 or not payload.is_contiguous():
            payload = payload.to(device=target_device, dtype=torch.int32).contiguous()
    except RuntimeError:
        # If .to(cuda) fails for any reason (unsupported dtype, OOM, etc.), use CPU fallback
        return _cpu_fallback_gemm(X, dense_payload)
    
    # Verify ALL tensors are actually on CUDA before launching kernel
    if not (exponents.is_cuda and signs.is_cuda and payload.is_cuda):
        return _cpu_fallback_gemm(X, dense_payload)
    
    if not X.is_contiguous():
        X = X.contiguous()
    
    M, K_x = X.shape
    R_padded, C_padded = dense_payload['padded_shape']
    C_orig, R_orig = dense_payload['orig_shape']
    
    if K_x < R_padded:
        X = torch.nn.functional.pad(X, (0, R_padded - K_x), mode='constant', value=0.0)
        
    Y = torch.empty((M, C_padded), device=target_device, dtype=X.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(C_padded, META['BLOCK_SIZE_N']),)
    
    # Wrap in CUDA device context manager to ensure Triton launches on the correct GPU
    with torch.cuda.device(target_device):
        oabf_gemm_kernel[grid](
            X, Y,
            exponents,
            signs,
            payload,
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
        self.register_buffer('_sp_value', sparse_outliers['value'].to(torch.float32).to(device))
        self._sparse_meta = {
            'count': sparse_outliers['count'],
        }
        
        # Pre-compute and cache index tensors for SpMV path to avoid recalculating on every forward pass
        if sparse_outliers['count'] > 0:
            block_idx_cpu = sparse_outliers['block_idx'].long()
            offset_cpu = sparse_outliers['offset'].long()
            padded_R, padded_C = dense_payload['padded_shape']
            C_orig, R_orig = dense_payload['orig_shape']
            
            global_idx_cpu = block_idx_cpu * 16 + offset_cpu
            row_idx_cpu = global_idx_cpu % padded_R
            col_idx_cpu = global_idx_cpu // padded_R
            
            # CPU-side boundary validation
            if (row_idx_cpu < 0).any() or (row_idx_cpu >= padded_R).any():
                raise IndexError(f"FlashBFP Compressor Error: Outlier row index out of padded bounds [0, {padded_R-1}]")
            if (col_idx_cpu < 0).any() or (col_idx_cpu >= padded_C).any():
                raise IndexError(f"FlashBFP Compressor Error: Outlier column index out of padded bounds [0, {padded_C-1}]")
                
            # Filter outliers belonging to valid active features
            valid_mask = (row_idx_cpu >= 0) & (row_idx_cpu < self.in_features) & \
                         (col_idx_cpu >= 0) & (col_idx_cpu < self.out_features)
                         
            r_idx_cpu = row_idx_cpu[valid_mask].long()
            c_idx_cpu = col_idx_cpu[valid_mask].long()
            v_val_cpu = sparse_outliers['value'][valid_mask].to(torch.float32)
            
            # Ironclad safety clamp
            if c_idx_cpu.numel() > 0:
                c_idx_cpu = torch.clamp(c_idx_cpu, 0, self.out_features - 1)
                r_idx_cpu = torch.clamp(r_idx_cpu, 0, self.in_features - 1)
                
            # Store cached GPU/device tensors as standard attributes (not registered buffers)
            self._sp_r_idx = r_idx_cpu.to(device)
            self._sp_c_idx = c_idx_cpu.to(device)
            self._sp_v_val = v_val_cpu.to(device)
        else:
            self._sp_r_idx = None
            self._sp_c_idx = None
            self._sp_v_val = None
        
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
        exec_device = self._dp_exponents.device
        
        # Explicitly move non-buffer attributes to exec_device if needed.
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
        
        # Fast Outlier SpMV Path using Cached GPU Indices
        if self._sparse_meta['count'] > 0 and self._sp_r_idx is not None:
            # Sync pre-computed attributes to exec_device if needed
            if self._sp_r_idx.device != exec_device:
                self._sp_r_idx = self._sp_r_idx.to(exec_device)
                self._sp_c_idx = self._sp_c_idx.to(exec_device)
                self._sp_v_val = self._sp_v_val.to(exec_device)
                
            r_idx = self._sp_r_idx
            c_idx = self._sp_c_idx
            v_val = self._sp_v_val.to(X_flat.dtype)
            
            if c_idx.numel() > 0:
                X_active = X_flat[:, r_idx]
                products = X_active * v_val[None, :]
                
                Y_sparse = torch.zeros_like(Y_dense)
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
