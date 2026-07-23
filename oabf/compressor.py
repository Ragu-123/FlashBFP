import jax
import jax.numpy as jnp
from typing import Tuple, Dict

class OABFCompressor:
    def __init__(self, block_size: int = 16, variance_threshold: float = 1e-5):
        self.block_size = block_size
        self.variance_threshold = variance_threshold

    def compute_dynamic_threshold(self, W: jax.Array) -> float:
        """
        Computes a dynamic outlier threshold based on layer-wise standard deviation (sigma)
        and Kurtosis to automatically adjust outlier rates across sub-networks.
        """
        mean = jnp.mean(W)
        variance = jnp.var(W)
        std_dev = jnp.sqrt(variance)
        
        # Compute Kurtosis (4th standardized moment)
        fourth_moment = jnp.mean((W - mean) ** 4)
        kurtosis = fourth_moment / (variance ** 2 + 1e-8)
        
        # Base factor derived from Shannon threshold bounds
        base_factor = jnp.sqrt(2.0 * jnp.log(W.size + 1e-8))
        
        # Adjust threshold based on sparsity and Kurtosis
        # Higher Kurtosis (peaky distributions with long tails) gets a lower threshold (more outliers)
        dynamic_scale = 0.05 * (1.0 + 0.1 * jnp.log(kurtosis + 1e-8))
        threshold = std_dev * base_factor * dynamic_scale
        
        # Clip to reasonable range
        return float(jnp.clip(threshold, 0.01, 0.15))

    def compress_matrix(self, W: jax.Array) -> Tuple[Dict, Dict]:
        """
        Compresses weight matrix W into dense blocks and a sparse outlier list.
        """
        orig_shape = W.shape
        W_flat = W.flatten()
        size = W_flat.size
        
        # Compute dynamic outlier threshold
        threshold = self.compute_dynamic_threshold(W)
        
        # Extract outliers
        outlier_mask = jnp.abs(W_flat) > threshold
        outlier_indices = jnp.where(outlier_mask)[0]
        outlier_values = W_flat[outlier_indices]
        
        # Zero out outliers in the dense matrix
        dense_W = jnp.where(outlier_mask, 0.0, W_flat)
        
        # Create Sparse Outliers structure (in COO block format)
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
        
        # Align elements to shared exponent and find required bitwidth W per block
        block_vars = jnp.var(blocks, axis=-1)
        bitwidths = jnp.clip(jnp.ceil(block_vars / self.variance_threshold), 2, 4).astype(jnp.int32)
        
        # Pack signs
        signs = jnp.sign(blocks) < 0
        
        dense_payload = {
            'exponents': shared_exponents,
            'bitwidths': bitwidths,
            'signs': signs,
            'blocks': blocks,  # Stored in compressed state in real system
            'orig_shape': orig_shape
        }
        
        return dense_payload, sparse_outliers
