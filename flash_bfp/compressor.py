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
        Returns the compressed dense payload (signs, exponents, packed mantissas)
        and the sparse outlier list in Coordinate format.
        """
        if W.numel() == 0:
            raise ValueError("Weight matrix cannot be empty.")
            
        orig_shape = W.shape
        C_orig, R_orig = orig_shape  # out_features, in_features
        
        # Pad rows (in_features) to multiple of block_size (16)
        remainder_r = R_orig % self.block_size
        padded_R = R_orig if remainder_r == 0 else R_orig + (self.block_size - remainder_r)
        
        # Pad columns (out_features) to multiple of tile_size (64)
        remainder_c = C_orig % self.tile_size
        padded_C = C_orig if remainder_c == 0 else C_orig + (self.tile_size - remainder_c)
        
        # Pad weight matrix
        pad_r = padded_R - R_orig
        pad_c = padded_C - C_orig
        W_padded = torch.nn.functional.pad(W, (0, pad_r, 0, pad_c), mode='constant', value=0.0)
        
        # Column-Major flattening to ensure contiguous GPU reading of column tiles
        # Shape: (padded_C, padded_R) -> W_padded.t() has shape (padded_R, padded_C)
        # Flattening W_padded.t() column-by-column means we flatten W_padded directly in row-major
        # because W_padded has shape (padded_C, padded_R).
        # Yes! Flat representation of Column-Major weight tiles is W_padded.flatten()
        W_flat = W_padded.flatten()
        size = W_flat.numel()
        
        # Compute threshold and extract outliers
        threshold = self.compute_dynamic_threshold(W)
        outlier_mask = torch.abs(W_flat) > threshold
        outlier_indices = torch.where(outlier_mask)[0]
        outlier_values = W_flat[outlier_indices]
        
        # Zero out outliers in the dense matrix
        dense_W = torch.where(outlier_mask, 0.0, W_flat)
        
        # Column-Major coordinate indices:
        # Since it is column-major, index represents column-major position:
        # row = index % padded_R, col = index // padded_R
        # W_padded shape is (padded_C, padded_R).
        # In flat row-major W_padded, the index matches row-major, which is:
        # row = index % padded_R, col = index // padded_R (since row dimension of W_padded is C)
        # Wait: W_padded index is index = col * padded_R + row.
        # So row = index % padded_R, col = index // padded_R is exactly column-major mapping!
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
        # (This is the real, physical 4-bit quantization step!)
        mantissas_int = torch.clamp(torch.round(blocks_aligned * 15.0), 0, 15).to(torch.int32)
        
        # Signs (True = negative, False = positive)
        signs = (blocks < 0)
        
        # --- BIT-PACKING ENGINE ---
        # Pack 16 signs into a single uint16 word
        signs_packed = torch.zeros(num_blocks, dtype=torch.int16, device=W.device)
        for i in range(16):
            signs_packed |= (signs[:, i].to(torch.int16) << i)
            
        # Pack sixteen 4-bit mantissas into two int32 words (each word holds 8 mantissas)
        payload = torch.zeros((num_blocks, 2), dtype=torch.int32, device=W.device)
        for i in range(8):
            payload[:, 0] |= (mantissas_int[:, i] << (i * 4))
            payload[:, 1] |= (mantissas_int[:, i + 8] << (i * 4))
            
        dense_payload = {
            'exponents': shared_exponents.to(torch.int8),
            'signs': signs_packed,
            'payload': payload,
            'orig_shape': orig_shape,
            'padded_shape': (padded_R, padded_C)
        }
        
        return dense_payload, sparse_outliers
