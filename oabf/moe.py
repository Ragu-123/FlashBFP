import jax
import jax.numpy as jnp
from typing import List, Tuple, Dict
from oabf.engine import OABFEngine

class OABFMoELayer:
    def __init__(self, block_size: int = 16):
        self.engine = OABFEngine(block_size=block_size)

    def run_moe_layer(self, X: jax.Array, experts_data: List[Tuple[Dict, Dict]], gate_logits: jax.Array) -> jax.Array:
        """
        Implements expert-level conditional decompression triggered post-routing.
        Only decompresses the active experts for the token batch, keeping others compressed in VRAM.
        """
        batch_size, seq_len, d_model = X.shape
        X_flat = X.reshape((-1, d_model))
        num_tokens = X_flat.shape[0]
        
        # Evaluate gating routing decisions
        gate_probs = jax.nn.softmax(gate_logits, axis=-1)  # (num_tokens, num_experts)
        top1_expert = jnp.argmax(gate_probs, axis=-1)       # (num_tokens,)
        
        # Process each expert conditionally
        output = jnp.zeros_like(X_flat)
        
        for i, (dense_payload, sparse_outliers) in enumerate(experts_data):
            # Check if this expert has any active tokens routed to it
            active_mask = (top1_expert == i)
            num_active = jnp.sum(active_mask)
            
            # Conditional execution check (Triton/CUDA dynamic kernel execution)
            def run_expert(X_active):
                # Run the fused tiled GEMM only for active tokens on this expert
                return self.engine.fused_tiled_gemm(X_active, dense_payload, sparse_outliers)
                
            def idle_expert(X_active):
                return jnp.zeros((X_active.shape[0], d_model))
                
            # Dynamic branch: if expert is inactive, bypass decompression completely
            # We filter active tokens using boolean masking (dynamic size)
            # For JAX/XLA compiling, we can pad/mask or use conditional branches
            # Here we model the conditional branch
            expert_tokens = X_flat[active_mask, :]
            expert_out = jax.lax.cond(
                num_active > 0,
                run_expert,
                idle_expert,
                expert_tokens
            )
            
            # Place outputs back into target indices
            output = output.at[active_mask, :].set(expert_out)
            
        return output.reshape((batch_size, seq_len, d_model))
