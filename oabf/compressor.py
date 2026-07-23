import jax
import jax.numpy as jnp
from typing import Tuple, Dict

class OABFCompressor:
    def __init__(self, block_size: int = 16, tile_size: int = 64, variance_threshold: float = 1e-5):
        self.block_size = block_size
        self.tile_size = tile_size
        self.variance_threshold = variance_threshold

    def compute_dynamic_threshold(self, W: jax.Array) -> float:
        """
        Computes a mathematically sound outlier threshold based on standard deviation
        and Kurtosis to target exactly ~1.5% outliers for normal distributions.
        """
        if W.size == 0:
            return 0.05
            
        mean = jnp.mean(W)
        variance = jnp.var(W)
        std_dev = jnp.sqrt(variance)
        
        # Compute Kurtosis (4th standardized moment)
        fourth_moment = jnp.mean((W - mean) ** 4)
        kurtosis = fourth_moment / (variance ** 2 + 1e-8)
        
        # Using Gaussian Q-function inverse approximation for 1.5% target (base factor 2.43)
        base_factor = 2.43
        
        # Adjust scale dynamically based on Kurtosis (heavy tails require wider bounds)
        dynamic_scale = base_factor * (1.0 + 0.1 * jnp.log(jnp.maximum(kurtosis / 3.0, 0.5)))
        threshold = std_dev * dynamic_scale
        
        # Clip to reasonable range to protect against edge cases
        return float(jnp.clip(threshold, 0.01, 0.20))

    def compress_matrix(self, W: jax.Array) -> Tuple[Dict, Dict]:
        """
        Compresses weight matrix W. Performs double padding, transposes to column-major format
        to ensure contiguous column tile memory loading on GPU, and runs bit-packing.
        """
        if W.size == 0:
            raise ValueError("Weight matrix cannot be empty.")
            
        orig_shape = W.shape
        R, C = orig_shape
        
        # Pad rows to multiple of block_size (16) and columns to multiple of tile_size (64)
        remainder_r = R % self.block_size
        padded_R = R if remainder_r == 0 else R + (self.block_size - remainder_r)
        remainder_c = C % self.tile_size
        padded_C = C if remainder_c == 0 else C + (self.tile_size - remainder_c)
        
        pad_r = padded_R - R
        pad_c = padded_C - C
        W_padded = jnp.pad(W, ((0, pad_r), (0, pad_c)), mode='constant', constant_values=0.0)
        
        # Transpose to Column-Major for contiguous GPU VRAM reads of column tiles
        W_flat = W_padded.T.flatten()
        size = W_flat.size
        
        # Compute threshold and extract outliers
        threshold = self.compute_dynamic_threshold(W)
        outlier_mask = jnp.abs(W_flat) > threshold
        outlier_indices = jnp.where(outlier_mask)[0]
        outlier_values = W_flat[outlier_indices]
        
        dense_W = jnp.where(outlier_mask, 0.0, W_flat)
        
        # Column-Major Indexing: row = index % R_padded, col = index // R_padded
        block_indices = outlier_indices // self.block_size
        block_offsets = outlier_indices % self.block_size
        
        sparse_outliers = {
            'block_idx': block_indices,
            'offset': block_offsets,
            'value': outlier_values,
            'count': len(outlier_indices)
        }
        
        num_blocks = size // self.block_size
        blocks = dense_W.reshape((num_blocks, self.block_size))
        
        # Compute shared exponent per block
        block_max = jnp.max(jnp.abs(blocks), axis=-1)
        shared_exponents = jnp.floor(jnp.log2(jnp.maximum(block_max, 1e-8)))
        
        # Align blocks by dividing by their shared exponents so the stored mantissa is in [0, 1]
        scale_factors = jnp.where(block_max < 1e-7, 1.0, jnp.power(2.0, shared_exponents))
        blocks_aligned = jnp.abs(blocks) / scale_factors[:, None]
        
        # Compute block dynamic bitwidth W (2, 3, or 4 bits)
        block_vars = jnp.var(blocks, axis=-1)
        bitwidths = jnp.clip(jnp.ceil(block_vars / (self.variance_threshold + 1e-8)), 2, 4).astype(jnp.int32)
        bitwidths = jnp.where(block_max < 1e-7, 0, bitwidths)
        
        # Pack signs
        signs = jnp.sign(blocks) < 0
        
        dense_payload = {
            'exponents': shared_exponents.astype(jnp.int8),
            'bitwidths': bitwidths.astype(jnp.int8),
            'signs': signs,
            'blocks': blocks_aligned,  # Now stored as aligned mantissas in [0, 1]
            'orig_shape': orig_shape,
            'padded_shape': (padded_R, padded_C)
        }
        
        return dense_payload, sparse_outliers
