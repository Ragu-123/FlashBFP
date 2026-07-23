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
    """
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_ptr = x_ptr + offs_am[:, None] * stride_am + (k * BLOCK_SIZE_K + offs_k)[None, :] * stride_ak
        a_tile = tl.load(a_ptr)
        
        w_tile = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
        
        for step in range(0, BLOCK_SIZE_K // 16):
            k_block = k * (BLOCK_SIZE_K // 16) + step
            block_idx = offs_bn[None, :] * (K // 16) + k_block
            
            exp = tl.load(exponents_ptr + block_idx).to(tl.float32)
            signs = tl.load(signs_ptr + block_idx).to(tl.int32)
            
            p_word1 = tl.load(payload_ptr + block_idx * 2)
            p_word2 = tl.load(payload_ptr + block_idx * 2 + 1)
            
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
    tl.store(c_ptr, accumulator.to(a_tile.dtype))


def oabf_gemm(X: torch.Tensor, dense_payload: Dict) -> torch.Tensor:
    """
    PyTorch wrapper that configures and launches the Triton fused OABF GEMM kernel.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is not installed. Fused GEMM execution is only supported on GPU.")
        
    assert X.is_cuda, "Input tensor X must be on CUDA."
    
    M, K_x = X.shape
    R_padded, C_padded = dense_payload['padded_shape']
    R_orig, C_orig = dense_payload['orig_shape']
    
    if K_x < R_padded:
        X = torch.nn.functional.pad(X, (0, R_padded - K_x), mode='constant', value=0.0)
        
    Y = torch.empty((M, C_padded), device=X.device, dtype=X.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(C_padded, META['BLOCK_SIZE_N']),)
    
    oabf_gemm_kernel[grid](
        X, Y,
        dense_payload['exponents'],
        dense_payload['signs'],
        dense_payload['payload'],
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
        self.dense_payload = {}
        self.sparse_outliers = {}
        
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def load_from_weight(self, W: torch.Tensor, compressor, device):
        """
        Compresses and loads weight matrix W into this linear layer on-the-fly.
        Compression runs entirely on the CPU to prevent device mismatch errors.
        """
        self.dense_payload, self.sparse_outliers = compressor.compress_matrix(W.cpu())
        
        # Send compressed tensors to the target device
        self.dense_payload['exponents'] = self.dense_payload['exponents'].to(device)
        self.dense_payload['signs'] = self.dense_payload['signs'].to(device)
        self.dense_payload['payload'] = self.dense_payload['payload'].to(device)
        
        self.sparse_outliers['block_idx'] = self.sparse_outliers['block_idx'].to(device)
        self.sparse_outliers['offset'] = self.sparse_outliers['offset'].to(device)
        self.sparse_outliers['value'] = self.sparse_outliers['value'].to(device)
        
        self.compressed = True

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Executes fused OABF Linear forward pass:
        1. Run Fused Triton GEMM (dense path).
        2. Run Sparse Outliers SpMV (sparse path).
        3. Add Bias.
        """
        if not self.compressed:
            raise RuntimeError("Weight matrix has not been compressed and loaded.")
            
        orig_shape = X.shape
        X_flat = X.view(-1, self.in_features)
        
        Y_dense = oabf_gemm(X_flat, self.dense_payload)
        
        if self.sparse_outliers['count'] > 0:
            R_padded, C_padded = self.dense_payload['padded_shape']
            
            outlier_block = self.sparse_outliers['block_idx']
            outlier_offset = self.sparse_outliers['offset']
            outlier_val = self.sparse_outliers['value']
            
            global_indices = outlier_block * 16 + outlier_offset
            row_indices = global_indices % R_padded
            col_indices = global_indices // R_padded
            
            valid_mask = (row_indices < self.in_features) & (col_indices < self.out_features)
            r_idx = row_indices[valid_mask]
            c_idx = col_indices[valid_mask]
            v_val = outlier_val[valid_mask].to(X.dtype)
            
            X_active = X_flat[:, r_idx]
            products = X_active * v_val[None, :]
            
            Y_sparse = torch.zeros_like(Y_dense)
            Y_sparse.index_add_(1, c_idx, products)
            
            Y_dense = Y_dense + Y_sparse
            
        if self.bias is not None:
            Y_dense = Y_dense + self.bias
            
        return Y_dense.view(*orig_shape[:-1], self.out_features)
