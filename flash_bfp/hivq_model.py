import torch
from flash_bfp.hivq import HIVQLinear

def compress_gemma_model_hivq(model):
    """
    Traverses the Gemma 4 model graph recursively and replaces all target linear projection
    layers with 2-bit HIVQLinear layers.
    """
    print("Beginning Gemma 4 model compression surgery using HIVQ...")
    
    # Track linear modules to replace
    linear_layers = []
    
    def find_linear_layers(module, name="model"):
        for sub_name, sub_module in module.named_children():
            full_name = f"{name}.{sub_name}"
            if isinstance(sub_module, torch.nn.Linear):
                # Compress only main compute-heavy attention and FFN layers.
                # Skip small/sensitive auxiliary layers (like lm_head, per-layer gates/projections,
                # vision/audio embedders and poolers) to preserve full model accuracy.
                is_target_layer = (
                    "self_attn" in full_name or
                    "mlp" in full_name or
                    "feed_forward" in full_name or
                    "lconv1d" in full_name
                )
                if not is_target_layer:
                    print(f"Skipping compression of auxiliary/sensitive layer: {full_name}")
                    continue
                linear_layers.append((module, sub_name, sub_module, full_name))
            else:
                find_linear_layers(sub_module, full_name)
                
    find_linear_layers(model)
    print(f"Found {len(linear_layers)} linear projection layers to compress.")
    
    from tqdm import tqdm
    
    pbar = tqdm(linear_layers, desc="Compressing layers to 2-bit E8", dynamic_ncols=True)
    for parent_module, sub_name, original_linear, full_name in pbar:
        # Update progress bar postfix with a clean, truncated layer name
        display_name = full_name.replace("model.model.language_model.", "")
        pbar.set_postfix_str(f"{display_name[:45]}")
        
        # Determine target device
        weight_device = original_linear.weight.device
        if weight_device.type == 'meta':
            target_device = torch.device('cpu')
        else:
            target_device = weight_device
            
        # Instantiate 2-bit HIVQ Linear Layer
        hivq_layer = HIVQLinear(
            in_features=original_linear.in_features,
            out_features=original_linear.out_features,
            bias=original_linear.bias is not None
        )
        
        # Copy bias if present
        if original_linear.bias is not None:
            bias_dev = original_linear.bias.device
            if bias_dev.type == 'meta':
                pass  # Skip copy for meta bias
            else:
                hivq_layer.bias.data.copy_(original_linear.bias.data)
                
        # Send layer to the target device immediately
        hivq_layer = hivq_layer.to(target_device)
        
        # Compress and load weight parameters on-the-fly
        # Materialize from meta if needed
        W = original_linear.weight.data
        if W.device.type == 'meta':
            W = torch.zeros(W.shape, dtype=W.dtype, device='cpu')
            
        hivq_layer.load_from_weight(W, device=target_device)
        
        # Replace sub-module inside the parent module
        setattr(parent_module, sub_name, hivq_layer)
        
        # Free original weight tensor immediately to reclaim VRAM
        del original_linear.weight
        if original_linear.bias is not None:
            del original_linear.bias
        del original_linear
        
        # Reclaim GPU memory cache
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
    print("\nGemma 4 model compression to 2-bit completed successfully! All target layers are now running HIVQ.")
    return model
