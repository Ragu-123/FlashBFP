import torch
from flash_bfp.hivq import HIVQLinear, HIVQEmbedding, HIVQTiedHead

def compress_gemma_model_hivq(model):
    """
    Traverses the Gemma 4 model graph recursively and replaces target linear projection
    layers and embedding layers with 2-bit HIVQ modules.
    
    Handles lm_head weight tying: when tie_word_embeddings=True (Gemma 4 default),
    lm_head is replaced with HIVQTiedHead instead of being independently compressed.
    """
    print("Beginning Gemma 4 model compression surgery using HIVQ...")
    
    # Check for weight tying
    tie_word_embeddings = getattr(model.config, 'tie_word_embeddings', False)
    if hasattr(model.config, 'text_config'):
        tie_word_embeddings = getattr(model.config.text_config, 'tie_word_embeddings', tie_word_embeddings)
    print(f"tie_word_embeddings = {tie_word_embeddings}")
    
    # Track modules to replace
    linear_layers = []
    embedding_layers = []
    lm_head_info = None  # Separate tracking for lm_head when tied
    
    def find_target_layers(module, name="model"):
        nonlocal lm_head_info
        for sub_name, sub_module in module.named_children():
            full_name = f"{name}.{sub_name}"
            if isinstance(sub_module, torch.nn.Linear):
                # Compress attention, MLP, logit heads, and per-layer embedding projections/gates
                is_target_layer = (
                    "self_attn" in full_name or
                    "mlp" in full_name or
                    "feed_forward" in full_name or
                    "lconv1d" in full_name or
                    "per_layer" in full_name or
                    "lm_head" in full_name
                )
                if not is_target_layer:
                    print(f"Skipping compression of auxiliary/sensitive layer: {full_name}")
                    continue
                    
                # If weight tying is enabled, handle lm_head separately
                if tie_word_embeddings and "lm_head" in full_name:
                    lm_head_info = (module, sub_name, sub_module, full_name)
                    print(f"Will use tied head for: {full_name}")
                    continue
                    
                linear_layers.append((module, sub_name, sub_module, full_name))
            elif isinstance(sub_module, torch.nn.Embedding) or "embedding" in sub_module.__class__.__name__.lower():
                # Compress target vocab embeddings
                is_target_emb = "embed_tokens" in full_name or "embedding" in full_name.lower()
                if is_target_emb:
                    embedding_layers.append((module, sub_name, sub_module, full_name))
            else:
                find_target_layers(sub_module, full_name)
                
    find_target_layers(model)
    lm_head_note = " (lm_head will use tied embedding)" if lm_head_info else ""
    print(f"Found {len(linear_layers)} linear layers and {len(embedding_layers)} embedding layers to compress.{lm_head_note}")
    
    from tqdm import tqdm
    
    # 1. Compress Linear Layers
    pbar_lin = tqdm(linear_layers, desc="Compressing linear layers", dynamic_ncols=True)
    for parent_module, sub_name, original_linear, full_name in pbar_lin:
        display_name = full_name.replace("model.model.language_model.", "")
        pbar_lin.set_postfix_str(f"{display_name[:45]}")
        
        weight_device = original_linear.weight.device
        if weight_device.type == 'meta':
            target_device = torch.device('cpu')
        else:
            target_device = weight_device
            
        hivq_layer = HIVQLinear(
            in_features=original_linear.in_features,
            out_features=original_linear.out_features,
            bias=original_linear.bias is not None
        )
        
        if original_linear.bias is not None:
            bias_dev = original_linear.bias.device
            if bias_dev.type == 'meta':
                pass
            else:
                hivq_layer.bias.data.copy_(original_linear.bias.data)
                
        hivq_layer = hivq_layer.to(target_device)
        
        W = original_linear.weight.data
        if W.device.type == 'meta':
            W = torch.zeros(W.shape, dtype=W.dtype, device='cpu')
            
        hivq_layer.load_from_weight(W, device=target_device)
        setattr(parent_module, sub_name, hivq_layer)
        
        del original_linear.weight
        if original_linear.bias is not None:
            del original_linear.bias
        del original_linear
        
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
    # 2. Compress Embedding Layers
    embed_tokens_hivq = None  # Track embed_tokens for lm_head tying
    pbar_emb = tqdm(embedding_layers, desc="Compressing embedding layers", dynamic_ncols=True)
    for parent_module, sub_name, original_emb, full_name in pbar_emb:
        display_name = full_name.replace("model.model.language_model.", "")
        pbar_emb.set_postfix_str(f"{display_name[:45]}")
        
        weight_device = original_emb.weight.device
        if weight_device.type == 'meta':
            target_device = torch.device('cpu')
        else:
            target_device = weight_device
            
        embed_scale = 1.0
        if hasattr(original_emb, "embed_scale"):
            embed_scale = original_emb.embed_scale.item()
        elif hasattr(original_emb, "scalar_embed_scale"):
            embed_scale = original_emb.scalar_embed_scale
            
        hivq_emb = HIVQEmbedding(
            num_embeddings=original_emb.num_embeddings,
            embedding_dim=original_emb.embedding_dim,
            padding_idx=original_emb.padding_idx,
            embed_scale=embed_scale
        )
        
        hivq_emb = hivq_emb.to(target_device)
        
        W = original_emb.weight.data
        if W.device.type == 'meta':
            W = torch.zeros(W.shape, dtype=W.dtype, device='cpu')
            
        hivq_emb.load_from_weight(W, device=target_device)
        setattr(parent_module, sub_name, hivq_emb)
        
        # Track the main embed_tokens for lm_head tying
        if "embed_tokens_per_layer" not in full_name and "embed_tokens" in full_name:
            embed_tokens_hivq = hivq_emb
        
        del original_emb.weight
        del original_emb
        
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    
    # 3. Handle lm_head weight tying
    if lm_head_info is not None and embed_tokens_hivq is not None:
        parent_module, sub_name, original_lm_head, full_name = lm_head_info
        print(f"Replacing lm_head with HIVQTiedHead (shares embed_tokens weight)...")
        tied_head = HIVQTiedHead(embed_tokens_hivq)
        setattr(parent_module, sub_name, tied_head)
        del original_lm_head
        import gc
        gc.collect()
        print("lm_head tied to embed_tokens successfully!")
    elif lm_head_info is not None:
        print("WARNING: lm_head found for tying but embed_tokens was not found. Compressing independently.")
        # Fall back to independent compression
        parent_module, sub_name, original_linear, full_name = lm_head_info
        weight_device = original_linear.weight.device
        target_device = torch.device('cpu') if weight_device.type == 'meta' else weight_device
        hivq_layer = HIVQLinear(
            in_features=original_linear.in_features,
            out_features=original_linear.out_features,
            bias=original_linear.bias is not None
        )
        hivq_layer = hivq_layer.to(target_device)
        W = original_linear.weight.data
        if W.device.type == 'meta':
            W = torch.zeros(W.shape, dtype=W.dtype, device='cpu')
        hivq_layer.load_from_weight(W, device=target_device)
        setattr(parent_module, sub_name, hivq_layer)
        del original_linear
        
    print("\nGemma 4 model compression to 2-bit completed successfully! All core layers (linear & embedding) are now running HIVQ.")
    return model


def load_gemma_model_hivq_skeleton(model):
    """
    Traverses the Gemma 4 model graph and replaces target linear and embedding layers
    with empty HIVQ modules (without running weight compression) so that a saved
    state dict can be loaded directly in 5 seconds.
    
    Handles lm_head weight tying: when tie_word_embeddings=True, lm_head is replaced
    with HIVQTiedHead that delegates to the embed_tokens HIVQ embedding.
    """
    print("Replacing layers with empty 2-bit HIVQ skeletons...")
    
    # Check for weight tying
    tie_word_embeddings = getattr(model.config, 'tie_word_embeddings', False)
    if hasattr(model.config, 'text_config'):
        tie_word_embeddings = getattr(model.config.text_config, 'tie_word_embeddings', tie_word_embeddings)
    
    linear_layers = []
    embedding_layers = []
    lm_head_info = None
    
    def find_target_layers(module, name="model"):
        nonlocal lm_head_info
        for sub_name, sub_module in module.named_children():
            full_name = f"{name}.{sub_name}"
            if isinstance(sub_module, torch.nn.Linear):
                is_target_layer = (
                    "self_attn" in full_name or
                    "mlp" in full_name or
                    "feed_forward" in full_name or
                    "lconv1d" in full_name or
                    "per_layer" in full_name or
                    "lm_head" in full_name
                )
                if is_target_layer:
                    if tie_word_embeddings and "lm_head" in full_name:
                        lm_head_info = (module, sub_name, sub_module, full_name)
                    else:
                        linear_layers.append((module, sub_name, sub_module, full_name))
            elif isinstance(sub_module, torch.nn.Embedding) or "embedding" in sub_module.__class__.__name__.lower():
                is_target_emb = "embed_tokens" in full_name or "embedding" in full_name.lower()
                if is_target_emb:
                    embedding_layers.append((module, sub_name, sub_module, full_name))
            else:
                find_target_layers(sub_module, full_name)
                
    find_target_layers(model)
    lm_head_note = " (lm_head will use tied embedding)" if lm_head_info else ""
    print(f"Swapping {len(linear_layers)} linear layers and {len(embedding_layers)} embedding layers with empty HIVQ modules...{lm_head_note}")
    
    # 1. Swap Linear Layers
    for parent_module, sub_name, original_linear, full_name in linear_layers:
        hivq_layer = HIVQLinear(
            in_features=original_linear.in_features,
            out_features=original_linear.out_features,
            bias=original_linear.bias is not None
        )
        if original_linear.bias is not None:
            hivq_layer.bias = torch.nn.Parameter(torch.zeros_like(original_linear.bias))
        setattr(parent_module, sub_name, hivq_layer)
        
    # 2. Swap Embedding Layers
    embed_tokens_hivq = None
    for parent_module, sub_name, original_emb, full_name in embedding_layers:
        embed_scale = 1.0
        if hasattr(original_emb, "embed_scale"):
            embed_scale = original_emb.embed_scale.item()
        elif hasattr(original_emb, "scalar_embed_scale"):
            embed_scale = original_emb.scalar_embed_scale
            
        hivq_emb = HIVQEmbedding(
            num_embeddings=original_emb.num_embeddings,
            embedding_dim=original_emb.embedding_dim,
            padding_idx=original_emb.padding_idx,
            embed_scale=embed_scale
        )
        setattr(parent_module, sub_name, hivq_emb)
        
        # Track embed_tokens for lm_head tying
        if "embed_tokens_per_layer" not in full_name and "embed_tokens" in full_name:
            embed_tokens_hivq = hivq_emb
    
    # 3. Handle lm_head weight tying
    if lm_head_info is not None and embed_tokens_hivq is not None:
        parent_module, sub_name, original_lm_head, full_name = lm_head_info
        tied_head = HIVQTiedHead(embed_tokens_hivq)
        setattr(parent_module, sub_name, tied_head)
        print(f"Replaced lm_head with HIVQTiedHead (tied to embed_tokens)")
    elif lm_head_info is not None:
        # Fallback: swap lm_head as normal HIVQLinear
        parent_module, sub_name, original_linear, full_name = lm_head_info
        hivq_layer = HIVQLinear(
            in_features=original_linear.in_features,
            out_features=original_linear.out_features,
            bias=original_linear.bias is not None
        )
        setattr(parent_module, sub_name, hivq_layer)
        
    print("Skeleton replacement completed successfully!")
    return model
