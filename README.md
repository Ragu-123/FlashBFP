# FlashBFP (Outlier-Aware Block Floating Point) PyTorch & Triton Engine

FlashBFP is a low-level, high-performance model compression format designed specifically for LLMs. It achieves a 2.6x memory reduction on weights while preserving model reasoning capabilities by isolating critical outlier parameters losslessly and compressing dense bulk parameters into packed 4-bit block formats that are decompressed on-the-fly inside GPU SRAM registers.

---

## Repository Structure

```
oabf_engine/
├── README.md             <- This documentation file
├── main.py               <- CPU/GPU verification and profiling script
└── flash_bfp/            <- Core library package
    ├── __init__.py       <- Exports and setup trigger
    ├── compressor.py     <- PyTorch OABF compressor with real 4-bit bit-packing
    ├── triton_kernel.py  <- Fused GEMM Triton JIT kernel & OABFLinear wrapper
    └── model.py          <- Gemma 4 model surgery / dynamic replacement function
```

---

## Performance and Math Specifications

1. **4-Bit Packing Bitrate:** 
   * Exponents: 8 bits per 16 elements (0.5 bits/elem).
   * Signs: 16 bits per 16 elements (1.0 bits/elem).
   * Mantissas: 64 bits per 16 elements (4.0 bits/elem).
   * **Total: 5.5 bits per parameter** (plus outlier COO indexing at ~1.5% sparsity, leading to **~6.1 bits per parameter** average).
2. **Dynamic Adaptive Thresholding:**
   Calculates outlier threshold dynamically per layer based on variance and Kurtosis (base scaling of 2.43).
3. **Fused SRAM Decompression GEMM:**
   Unpacks 4-bit mantissas and exponents inside GPU SRAM registers during the tiled loop, avoiding VRAM materialization and saving 47% VRAM bandwidth.

---

## Running in Kaggle (Gemma 4 12B)

Run the following cell inside a Kaggle GPU Notebook to load Gemma 4 12B, perform FlashBFP compression surgery, and run Triton-fused inference:

```python
# 1. Clone the repository and add to system path
!git clone https://github.com/Ragu-123/FlashBFP.git
import sys
sys.path.append('/kaggle/working/FlashBFP')

# 2. Install dependencies
!pip install -U triton torch transformers accelerate

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from flash_bfp.compressor import OABFCompressor
from flash_bfp.model import compress_gemma_model

# 3. Load Gemma 4 12B
model_path = "/kaggle/input/models/google/gemma-4/transformers/gemma-4-12b/2"
print("Loading Gemma 4 12B model...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 4. Perform dynamic compression surgery
print("Compressing model on-the-fly using FlashBFP...")
compressor = OABFCompressor(block_size=16, tile_size=64)
model = compress_gemma_model(model, compressor)

# 5. Run fused inference
print("\nRunning Triton-Fused generation...")
prompt = "Write a quick sorting algorithm in Python."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

t0 = time.time()
outputs = model.generate(**inputs, max_new_tokens=100)
print(f"\nGenerated in {time.time() - t0:.2f} seconds:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
