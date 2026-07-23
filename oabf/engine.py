import jax
import jax.numpy as jnp
from typing import Dict

class OABFEngine:
    def __init__(self, block_size: int = 16, tile_size: int = 64):
        self.block_size = block_size
        self.tile_size = tile_size

    def fused_tiled_gemm(self, X: jax.Array, dense_payload: Dict, sparse_outliers: Dict) -> jax.Array:
        """
        Runs memory-fused SRAM decompression GEMM.
        Robustly handles padding for arbitrary shapes and zero-outlier edge cases.
        """
        # Original and padded shapes
        R_orig, C_orig = dense_payload['orig_shape']
        R_padded, C_padded = dense_payload['padded_shape']
        
        M, K_x = X.shape
        
        # Edge Case 1: Pad input activation X to match padded weight rows (R_padded)
        if K_x < R_padded:
            X = jnp.pad(X, ((0, 0), (0, R_padded - K_x)), mode='constant', constant_values=0.0)
            
        # Unpack dense payload
        exponents = dense_payload['exponents']
        bitwidths = dense_payload['bitwidths']
        signs = dense_payload['signs']
        blocks = dense_payload['blocks']
        
        num_tiles = C_padded // self.tile_size
        
        # Final output accumulator (for padded dimension)
        Y = jnp.zeros((M, C_padded))
        
        # Calculate block-level parameters
        num_blocks_tile = (self.tile_size * R_padded) // self.block_size
        
        # Loop over output tiles (simulating GPU thread block tiling)
        def tile_step(carry, tile_idx):
            X_in, Y_acc = carry
            col_start = tile_idx * self.tile_size
            
            # 1. SRAM Load and Reconstruction (Tiled dynamic slice)
            start_block_idx = (col_start * R_padded) // self.block_size
            
            # Load compressed blocks for this tile from VRAM to SRAM
            tile_blocks = jax.lax.dynamic_slice(blocks, (start_block_idx, 0), (num_blocks_tile, self.block_size))
            tile_exponents = jax.lax.dynamic_slice(exponents, (start_block_idx,), (num_blocks_tile,))
            tile_signs = jax.lax.dynamic_slice(signs, (start_block_idx, 0), (num_blocks_tile, self.block_size))
            
            # Reconstruct FP values in shared memory: (-1)^S * M * 2^Es
            tile_reconstructed_flat = jnp.where(tile_signs, -1.0, 1.0) * tile_blocks * jnp.power(2.0, tile_exponents[:, None])
            
            # Format back to 2D column tile: (R_padded, tile_size)
            W_dense_tile = tile_reconstructed_flat.reshape((self.tile_size, R_padded)).T
            
            # GEMM Inner Loop Accumulate
            Y_tile_dense = jnp.dot(X_in, W_dense_tile)
            
            # 2. Sparse Outlier SpMV Accumulation (with conditional bypass for 0 outliers)
            def run_spmv():
                outlier_val = sparse_outliers['value']
                outlier_block = sparse_outliers['block_idx']
                outlier_offset = sparse_outliers['offset']
                
                global_indices = outlier_block * self.block_size + outlier_offset
                row_indices = global_indices % R_padded
                col_indices = global_indices // R_padded
                
                # Filter outliers belonging to this column tile
                tile_mask = (col_indices >= col_start) & (col_indices < col_start + self.tile_size)
                tile_outlier_rows = jnp.where(tile_mask, row_indices, -1)
                tile_outlier_cols = jnp.where(tile_mask, col_indices - col_start, -1)
                tile_outlier_vals = jnp.where(tile_mask, outlier_val, 0.0)
                
                # Initialize sparse accumulator for this tile
                Y_tile_sp = jnp.zeros((M, self.tile_size))
                
                def accumulate_outliers(i, carry_sparse):
                    r = tile_outlier_rows[i]
                    c = tile_outlier_cols[i]
                    val = tile_outlier_vals[i]
                    
                    updated = carry_sparse.at[:, c].add(X_in[:, r] * val)
                    return jax.lax.cond(r != -1, lambda: updated, lambda: carry_sparse)
                    
                return jax.lax.fori_loop(0, sparse_outliers['count'], accumulate_outliers, Y_tile_sp)
                
            def idle_spmv():
                return jnp.zeros((M, self.tile_size))
                
            # Edge Case 3: Zero Outliers Check
            Y_tile_sparse = jax.lax.cond(sparse_outliers['count'] > 0, run_spmv, idle_spmv)
            
            # Fuse dense and sparse outputs inside registers
            Y_tile_fused = Y_tile_dense + Y_tile_sparse
            
            # Write final fused tile to VRAM once
            Y_acc = jax.lax.dynamic_update_slice(Y_acc, Y_tile_fused, (0, col_start))
            
            return (X_in, Y_acc), None
            
        _, Y_final = jax.lax.scan(tile_step, (X, Y), jnp.arange(num_tiles))
        
        # Edge Case 4: Crop the final output back to original column dimension (C_orig)
        return Y_final[:, :C_orig]
