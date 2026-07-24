import torch
from flash_bfp.compressor import OABFCompressor
from flash_bfp.triton_kernel import OABFLinear

def compress_gemma_model(model: torch.nn.Module, compressor: OABFCompressor) -> torch.nn.Module:
    """
    Performs surgery on the Gemma 4 PyTorch model to replace all standard nn.Linear layers
    with OABFLinear layers, compressing their weights on-the-fly.
    Bypasses embeddings and normalization layers to preserve full model reasoning capabilities.
    """
    print("Beginning Gemma 4 model compression surgery...")
    
    # Track linear modules to replace
    linear_layers = []
    
    def find_linear_layers(module, name="model"):
        for sub_name, sub_module in module.named_children():
            full_name = f"{name}.{sub_name}"
            if isinstance(sub_module, torch.nn.Linear):
                # Avoid modifying output projection or specific layers if requested,
                # but standard practice is to compress all QKV, projection, and FFN layers.
                linear_layers.append((module, sub_name, sub_module, full_name))
            else:
                find_linear_layers(sub_module, full_name)
                
    find_linear_layers(model)
    print(f"Found {len(linear_layers)} linear projection layers to compress.")
    
    from tqdm import tqdm
    
    pbar = tqdm(linear_layers, desc="Compressing layers", dynamic_ncols=True)
    for parent_module, sub_name, original_linear, full_name in pbar:
        # Update progress bar postfix with a clean, truncated layer name
        display_name = full_name.replace("model.model.language_model.", "")
        pbar.set_postfix_str(f"{display_name[:45]}")
        
        # Instantiate OABF Linear Layer
        oabf_layer = OABFLinear(
            in_features=original_linear.in_features,
            out_features=original_linear.out_features,
            bias=original_linear.bias is not None
        )
        
        # Copy bias if present
        if original_linear.bias is not None:
            oabf_layer.bias.data.copy_(original_linear.bias.data)
            
        # Send oabf_layer to the target device immediately to support multi-GPU sharding
        oabf_layer = oabf_layer.to(original_linear.weight.device)
        # Compress and load weight parameters on-the-fly
        # Weight in nn.Linear is stored as (out_features, in_features)
        oabf_layer.load_from_weight(original_linear.weight.data, compressor, device=original_linear.weight.device)
        
        # Replace sub-module inside the parent module
        setattr(parent_module, sub_name, oabf_layer)
        
        # Free original weight tensor immediately to reclaim VRAM
        del original_linear.weight
        if original_linear.bias is not None:
            del original_linear.bias
        del original_linear
        
        # Reclaim GPU memory cache to prevent VRAM accumulation
        import gc
        gc.collect()
        torch.cuda.empty_cache()
            
    print("\nGemma 4 model compression completed successfully! All linear layers are now running Fused Triton kernels.")
    return model
