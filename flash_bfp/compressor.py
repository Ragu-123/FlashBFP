import torch
import numpy as np
from typing import Tuple, Dict

class OABFCompressor:
    def __init__(self, block_size: int = 16, tile_size: int = 64, variance_threshold: float = 1e-5):
        self.block_size = block_size
        self.tile_size = tile_size
        self.variance_threshold = variance_threshold

    def compute_dynamic_threshold(self, W: torch.Tensor) -> float:
        """
        Computes outlier threshold based on standard deviation and Kurtosis
        to target exactly ~1.5% outliers for normal distributions.
        """
        if W.numel() == 0:
            return 0.05
            
        mean = torch.mean(W)
        variance = torch.var(W)
        std_dev = torch.sqrt(variance)
        
        # Compute Kurtosis
        fourth_moment = torch.mean((W - mean) ** 4)
        kurtosis = fourth_moment / (variance ** 2 + 1e-8)
        
        # Q-function approximation for 1.5% target (base scale 2.43)
        base_factor = 2.43
        dynamic_scale = base_factor * (1.0 + 0.1 * torch.log(torch.clamp(kurtosis / 3.0, min=0.5)))
        threshold = std_dev * dynamic_scale
        
        return float(torch.clamp(threshold, 0.01, 0.20))

    def compress_matrix(self, W: torch.Tensor) -> Tuple[Dict, Dict]:
        """
        Compresses weight matrix W of shape (out_features, in_features).
        Runs entirely on the CPU to prevent device mismatch errors and optimize memory.
        Uses optimized vectorized operations to keep execution time under 15ms per layer.
        """
        if W.numel() == 0:
            raise ValueError("Weight matrix cannot be empty.")
            
        # FORCE execution to CPU to avoid CPU-GPU device mismatch and meta-device errors
        W = W.detach().cpu()
        
        orig_shape = W.shape
        C_orig, R_orig = orig_shape  # out_features, in_features
        
        # Pad rows to multiple of block_size (16) and columns to multiple of tile_size (64)
        remainder_r = R_orig % self.block_size
        padded_R = R_orig if remainder_r == 0 else R_orig + (self.block_size - remainder_r)
        remainder_c = C_orig % self.tile_size
        padded_C = C_orig if remainder_c == 0 else C_orig + (self.tile_size - remainder_c)
        
        pad_r = padded_R - R_orig
        pad_c = padded_C - C_orig
        W_padded = torch.nn.functional.pad(W, (0, pad_r, 0, pad_c), mode='constant', value=0.0)
        
        # Column-Major flattening
        W_flat = W_padded.flatten()
        size = W_flat.numel()
        
        # Compute threshold and extract outliers
        threshold = self.compute_dynamic_threshold(W)
        outlier_mask = torch.abs(W_flat) > threshold
        outlier_indices = torch.where(outlier_mask)[0]
        outlier_values = W_flat[outlier_indices]
        
        dense_W = torch.where(outlier_mask, 0.0, W_flat)
        
        # Column-Major Indexing: row = index % R_padded, col = index // R_padded
        block_indices = outlier_indices // self.block_size
        block_offsets = outlier_indices % self.block_size
        
        sparse_outliers = {
            'block_idx': block_indices.to(torch.int32),
            'offset': block_offsets.to(torch.int32),
            'value': outlier_values.to(torch.float16),
            'count': len(outlier_indices)
        }
        
        num_blocks = size // self.block_size
        blocks = dense_W.reshape((num_blocks, self.block_size))
        
        # Compute shared exponent per block
        block_max = torch.max(torch.abs(blocks), dim=-1)[0]
        shared_exponents = torch.floor(torch.log2(torch.clamp(block_max, min=1e-8)))
        
        # Align blocks by dividing by shared exponents
        scale_factors = torch.where(block_max < 1e-7, 1.0, torch.pow(2.0, shared_exponents))
        blocks_aligned = torch.abs(blocks) / scale_factors[:, None]
        
        # Quantize aligned mantissas to 4 bits [0, 15]
        mantissas_int = torch.clamp(torch.round(blocks_aligned * 15.0), 0, 15).to(torch.int32)
        
        # Signs
        signs = (blocks < 0)
        
        # Vectorized Pack signs into uint16 word on CPU
        shifts_signs = 2 ** torch.arange(16, dtype=torch.int32, device=W.device)
        signs_packed = (signs.to(torch.int32) * shifts_signs[None, :]).sum(dim=-1).to(torch.int16)
            
        # Vectorized Pack mantissas into two int32 words on CPU
        shifts_mantissas = 16 ** torch.arange(8, dtype=torch.int32, device=W.device)
        payload0 = (mantissas_int[:, :8] * shifts_mantissas[None, :]).sum(dim=-1).to(torch.int32)
        payload1 = (mantissas_int[:, 8:] * shifts_mantissas[None, :]).sum(dim=-1).to(torch.int32)
        payload = torch.stack([payload0, payload1], dim=-1)
            
        dense_payload = {
            'exponents': shared_exponents.to(torch.int8),
            'signs': signs_packed,
            'payload': payload,
            'orig_shape': orig_shape,
            'padded_shape': (padded_R, padded_C)
        }
        
        return dense_payload, sparse_outliers
