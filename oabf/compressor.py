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
        Computes a dynamic outlier threshold based on layer-wise standard deviation (sigma)
        and Kurtosis to automatically adjust outlier rates across sub-networks.
        """
        if W.size == 0:
            return 0.05
            
        mean = jnp.mean(W)
        variance = jnp.var(W)
        std_dev = jnp.sqrt(variance)
        
        # Compute Kurtosis (4th standardized moment)
        fourth_moment = jnp.mean((W - mean) ** 4)
        kurtosis = fourth_moment / (variance ** 2 + 1e-8)
        
        # Base factor derived from Shannon threshold bounds
        base_factor = jnp.sqrt(2.0 * jnp.log(W.size + 1e-8))
        
        # Adjust threshold based on sparsity and Kurtosis
        dynamic_scale = 0.05 * (1.0 + 0.1 * jnp.log(kurtosis + 1e-8))
        threshold = std_dev * base_factor * dynamic_scale
        
        # Clip to reasonable range
        return float(jnp.clip(threshold, 0.01, 0.15))

    def compress_matrix(self, W: jax.Array) -> Tuple[Dict, Dict]:
        """
        Compresses weight matrix W into dense blocks and a sparse outlier list.
        Pads the matrix dimensions to be multiples of block_size (rows) and tile_size (cols)
        to handle arbitrary shapes robustly.
        """
        if W.size == 0:
            raise ValueError("Weight matrix cannot be empty.")
            
        orig_shape = W.shape
        R, C = orig_shape
        
        # Edge Case 1: Arbitrary shapes padding
        # Pad rows to multiple of block_size (16)
        remainder_r = R % self.block_size
        padded_R = R if remainder_r == 0 else R + (self.block_size - remainder_r)
        
        # Pad columns to multiple of tile_size (64)
        remainder_c = C % self.tile_size
        padded_C = C if remainder_c == 0 else C + (self.tile_size - remainder_c)
        
        # Perform padding with constant zeros
        pad_r = padded_R - R
        pad_c = padded_C - C
        W_padded = jnp.pad(W, ((0, pad_r), (0, pad_c)), mode='constant', constant_values=0.0)
        
        W_flat = W_padded.flatten()
        size = W_flat.size
        
        # Compute dynamic outlier threshold on original matrix to avoid padding skew
        threshold = self.compute_dynamic_threshold(W)
        
        # Extract outliers
        outlier_mask = jnp.abs(W_flat) > threshold
        outlier_indices = jnp.where(outlier_mask)[0]
        outlier_values = W_flat[outlier_indices]
        
        # Zero out outliers in the dense matrix
        dense_W = jnp.where(outlier_mask, 0.0, W_flat)
        
        # Create Sparse Outliers structure
        block_indices = outlier_indices // self.block_size
        block_offsets = outlier_indices % self.block_size
        
        sparse_outliers = {
            'block_idx': block_indices,
            'offset': block_offsets,
            'value': outlier_values,
            'count': len(outlier_indices)
        }
        
        # Reshape dense matrix into blocks
        num_blocks = size // self.block_size
        blocks = dense_W.reshape((num_blocks, self.block_size))
        
        # Compute shared exponent per block (log2 representation)
        block_max = jnp.max(jnp.abs(blocks), axis=-1)
        shared_exponents = jnp.floor(jnp.log2(jnp.maximum(block_max, 1e-8)))
        
        # Edge Case 2: Zero-variance or Constant Weight blocks
        block_vars = jnp.var(blocks, axis=-1)
        bitwidths = jnp.clip(jnp.ceil(block_vars / (self.variance_threshold + 1e-8)), 2, 4).astype(jnp.int32)
        # If block contains only zeros, set bitwidth to 0
        is_zero_block = block_max < 1e-7
        bitwidths = jnp.where(is_zero_block, 0, bitwidths)
        
        # Pack signs
        signs = jnp.sign(blocks) < 0
        
        dense_payload = {
            'exponents': shared_exponents,
            'bitwidths': bitwidths,
            'signs': signs,
            'blocks': blocks,
            'orig_shape': orig_shape,
            'padded_shape': (padded_R, padded_C)
        }
        
        return dense_payload, sparse_outliers
