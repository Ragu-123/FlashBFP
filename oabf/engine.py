import jax
import jax.numpy as jnp
from typing import Dict

class OABFEngine:
    def __init__(self, block_size: int = 16):
        self.block_size = block_size

    def fused_tiled_gemm(self, X: jax.Array, dense_payload: Dict, sparse_outliers: Dict) -> jax.Array:
        """
        Simulates the memory-fused SRAM decompression GEMM.
        Decompresses weight blocks tile-by-tile inside the execution loop.
        Weights are never materialized globally in VRAM.
        """
        # Dimensions
        M, K_x = X.shape
        K_w, N = dense_payload['orig_shape']
        
        # Unpack dense payload
        exponents = dense_payload['exponents']
        bitwidths = dense_payload['bitwidths']
        signs = dense_payload['signs']
        blocks = dense_payload['blocks']
        
        # We loop over tiles of the weight matrix along the output dimension (N)
        tile_size = 64
        num_tiles = N // tile_size
        
        # Final output accumulator
        Y = jnp.zeros((M, N))
        
        # Calculate block-level parameters
        num_blocks_tile = (tile_size * K_w) // self.block_size
        
        # Loop over output tiles (simulating GPU thread block tiling)
        def tile_step(carry, tile_idx):
            X_in, Y_acc = carry
            col_start = tile_idx * tile_size
            
            # 1. SRAM Load and Reconstruction (Tiled dynamic slice)
            # col_start * K_w gives the flattened start index
            start_block_idx = (col_start * K_w) // self.block_size
            
            # Load compressed blocks for this tile from VRAM to SRAM
            tile_blocks = jax.lax.dynamic_slice(blocks, (start_block_idx, 0), (num_blocks_tile, self.block_size))
            tile_exponents = jax.lax.dynamic_slice(exponents, (start_block_idx,), (num_blocks_tile,))
            tile_signs = jax.lax.dynamic_slice(signs, (start_block_idx, 0), (num_blocks_tile, self.block_size))
            
            # Reconstruct FP values in shared memory: (-1)^S * M * 2^Es
            # (In actual hardware, this is vector bit-shifting on registers)
            tile_reconstructed_flat = jnp.where(tile_signs, -1.0, 1.0) * tile_blocks * jnp.power(2.0, tile_exponents[:, None])
            
            # Format back to 2D column tile: (K_w, tile_size)
            # The columns are stacked, so we reshape and transpose
            W_dense_tile = tile_reconstructed_flat.reshape((tile_size, K_w)).T
            
            # GEMM Inner Loop Accumulate
            Y_tile_dense = jnp.dot(X_in, W_dense_tile)
            
            # 2. Sparse Outlier SpMV Accumulation
            outlier_val = sparse_outliers['value']
            outlier_block = sparse_outliers['block_idx']
            outlier_offset = sparse_outliers['offset']
            
            global_indices = outlier_block * self.block_size + outlier_offset
            row_indices = global_indices % K_w
            col_indices = global_indices // K_w
            
            # Filter outliers belonging to this column tile
            tile_mask = (col_indices >= col_start) & (col_indices < col_start + tile_size)
            tile_outlier_rows = jnp.where(tile_mask, row_indices, -1)
            tile_outlier_cols = jnp.where(tile_mask, col_indices - col_start, -1)
            tile_outlier_vals = jnp.where(tile_mask, outlier_val, 0.0)
            
            # Initialize sparse accumulator for this tile
            Y_tile_sparse = jnp.zeros((M, tile_size))
            
            # Parallel accumulate outlier contributions to the tile output
            # (Simulates fused SpMV accumulate inside register cache)
            def accumulate_outliers(i, carry_sparse):
                r = tile_outlier_rows[i]
                c = tile_outlier_cols[i]
                val = tile_outlier_vals[i]
                
                # If outlier is valid for this tile, add X[:, r] * val to Y[:, c]
                updated = carry_sparse.at[:, c].add(X_in[:, r] * val)
                return jax.lax.cond(r != -1, lambda: updated, lambda: carry_sparse)
                
            Y_tile_sparse = jax.lax.fori_loop(0, sparse_outliers['count'], accumulate_outliers, Y_tile_sparse)
            
            # Fuse dense and sparse outputs inside registers
            Y_tile_fused = Y_tile_dense + Y_tile_sparse
            
            # Write final fused tile to VRAM once
            # This uses dynamic slice update
            Y_acc = jax.lax.dynamic_update_slice(Y_acc, Y_tile_fused, (0, col_start))
            
            return (X_in, Y_acc), None
            
        _, Y_final = jax.lax.scan(tile_step, (X, Y), jnp.arange(num_tiles))
        return Y_final
