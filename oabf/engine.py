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
        Addresses bottlenecks:
        1. Fused SRAM decompression: uses tensordot directly on blocked layouts to avoid VRAM materialization.
        2. Parallel GPU SpMV: uses jax.ops.segment_sum to completely replace sequential loops.
        3. Memory Corruption Protection: redirects out-of-bounds tiles to a dummy segment.
        """
        # Original and padded shapes
        R_orig, C_orig = dense_payload['orig_shape']
        R_padded, C_padded = dense_payload['padded_shape']
        
        M, K_x = X.shape
        
        # Pad input activation X to match padded weight rows (R_padded)
        if K_x < R_padded:
            X = jnp.pad(X, ((0, 0), (0, R_padded - K_x)), mode='constant', constant_values=0.0)
            
        # Reshape X to block layout: (M, R_padded // 16, 16)
        num_blocks_K = R_padded // self.block_size
        X_blocked = X.reshape((M, num_blocks_K, self.block_size))
        
        # Unpack dense payload
        exponents = dense_payload['exponents']
        signs = dense_payload['signs']
        blocks = dense_payload['blocks']
        
        num_tiles = C_padded // self.tile_size
        
        # Final output accumulator (for padded dimension)
        Y = jnp.zeros((M, C_padded))
        
        # Calculate block-level parameters
        num_blocks_tile = (self.tile_size * R_padded) // self.block_size
        
        # Loop over output tiles (simulating GPU thread block tiling)
        def tile_step(carry, tile_idx):
            X_in, X_in_blocked, Y_acc = carry
            col_start = tile_idx * self.tile_size
            
            # 1. SRAM Load and Reconstruction (Tiled dynamic slice)
            # col_start * R_padded is the column-major start offset
            start_block_idx = (col_start * R_padded) // self.block_size
            
            # Load compressed blocks for this tile from VRAM to SRAM
            tile_blocks = jax.lax.dynamic_slice(blocks, (start_block_idx, 0), (num_blocks_tile, self.block_size))
            tile_exponents = jax.lax.dynamic_slice(exponents, (start_block_idx,), (num_blocks_tile,))
            tile_signs = jax.lax.dynamic_slice(signs, (start_block_idx, 0), (num_blocks_tile, self.block_size))
            
            # Reconstruct FP values in shared memory: (-1)^S * M * 2^Es
            tile_reconstructed_flat = jnp.where(tile_signs, -1.0, 1.0) * tile_blocks * jnp.power(2.0, tile_exponents[:, None])
            
            # Format directly to block layout for tensordot: (tile_size, R_padded // 16, 16)
            tile_reconstructed = tile_reconstructed_flat.reshape((self.tile_size, num_blocks_K, self.block_size))
            
            # Fused SRAM GEMM: Multiply directly on the blocked layout (no VRAM materialization)
            Y_tile_dense = jnp.tensordot(X_in_blocked, tile_reconstructed, axes=((1, 2), (1, 2)))
            
            # 2. Parallel GPU Outlier SpMV Accumulation (GPU Parallelism Preserved)
            def run_spmv():
                outlier_val = sparse_outliers['value']
                outlier_block = sparse_outliers['block_idx']
                outlier_offset = sparse_outliers['offset']
                
                # Column-Major Indexing: row = index % R_padded, col = index // R_padded
                global_indices = outlier_block * self.block_size + outlier_offset
                row_indices = global_indices % R_padded
                col_indices = global_indices // R_padded
                
                # Filter outliers belonging to this column tile
                tile_mask = (col_indices >= col_start) & (col_indices < col_start + self.tile_size)
                
                # GPU Parallelism: Gather input activations for active outlier rows in parallel
                # Map invalid outliers to row 0 and scale by 0.0 to prevent pollution
                safe_rows = jnp.where(tile_mask, row_indices, 0)
                X_active = X_in[:, safe_rows]  # Shape (M, N_out)
                
                # Scale by outlier values (zeroing out inactive ones)
                tile_outlier_vals = jnp.where(tile_mask, outlier_val, 0.0)
                products = X_active * tile_outlier_vals[None, :]  # Shape (M, N_out)
                
                # Redirect invalid outliers to a dummy segment (index: tile_size) to prevent memory corruption
                safe_cols = jnp.where(tile_mask, col_indices - col_start, self.tile_size)
                
                # Transpose to shape (N_out, M) to segment over leading axis 0
                products_T = products.T
                
                # Run parallel segment sum over leading axis
                Y_tile_sp_T = jax.ops.segment_sum(
                    data=products_T,
                    segment_ids=safe_cols,
                    num_segments=self.tile_size + 1
                )
                
                # Transpose back and slice off the dummy column at index tile_size
                return Y_tile_sp_T.T[:, :self.tile_size]
                
            def idle_spmv():
                return jnp.zeros((M, self.tile_size))
                
            # Zero Outliers Check
            Y_tile_sparse = jax.lax.cond(sparse_outliers['count'] > 0, run_spmv, idle_spmv)
            
            # Fuse dense and sparse outputs inside SRAM registers
            Y_tile_fused = Y_tile_dense + Y_tile_sparse
            
            # Write final fused tile to VRAM once
            Y_acc = jax.lax.dynamic_update_slice(Y_acc, Y_tile_fused, (0, col_start))
            
            return (X_in, X_in_blocked, Y_acc), None
            
        (_, _, Y_final), _ = jax.lax.scan(tile_step, (X, X_blocked, Y), jnp.arange(num_tiles))
        
        # Crop the final output back to original column dimension (C_orig)
        return Y_final[:, :C_orig]
