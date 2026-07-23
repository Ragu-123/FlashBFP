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
        Uses static shapes to prevent JAX JIT NonConcreteBooleanIndexError tracing failures.
        """
        batch_size, seq_len, d_model = X.shape
        X_flat = X.reshape((-1, d_model))
        
        # Evaluate gating routing decisions
        gate_probs = jax.nn.softmax(gate_logits, axis=-1)  # (num_tokens, num_experts)
        top1_expert = jnp.argmax(gate_probs, axis=-1)       # (num_tokens,)
        
        # Process each expert conditionally
        output = jnp.zeros_like(X_flat)
        
        for i, (dense_payload, sparse_outliers) in enumerate(experts_data):
            # Static Shape Masking: Avoid boolean index slicing which creates dynamic shapes
            active_mask = (top1_expert == i).astype(jnp.float32)
            
            # Multiply input activations by mask (zeroes out inactive tokens, keeping shape static)
            expert_tokens = X_flat * active_mask[:, None]
            
            # Execute fused tiled GEMM on static shape
            expert_out = self.engine.fused_tiled_gemm(expert_tokens, dense_payload, sparse_outliers)
            
            # Accumulate back only active expert outputs (mask-gated)
            output = output + (expert_out * active_mask[:, None])
            
        return output.reshape((batch_size, seq_len, d_model))
