# Data/Scripts

This directory contains helper scripts for preparing model weights and other data assets.

# Installation

To set up the environment for running the scripts, follow these steps:

1. Open a Developer command shell and change to the scripts directory:
cd Data\Scripts

2. Create and activate a virtual environment, then install dependencies:
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
## Install CPU-only torch first to avoid large GPU wheels (optional but recommended)
pip install torch --index-url https://download.pytorch.org/whl/cpu
## Then install remaining requirements
pip install -r requirements.txt

# Running scripts

- All scripts should be executed from the `Data\Scripts` directory with the virtual environment activated.

- convert_gpt2.py (example usage)
- Purpose: convert HuggingFace GPT-2 weights to the Mila binary format.
- Examples:
 ```powershell
 python convert_gpt2.py --model gpt2 --output ../Weights/gpt2/gpt2_small.bin
 python convert_gpt2.py --model gpt2-medium --output ../Weights/gpt2/gpt2_medium.bin
 python convert_gpt2.py --model gpt2-large --output ../Weights/gpt2/gpt2_large.bin
 python convert_gpt2.py --model gpt2-xl --output ../Weights/gpt2/gpt2_xl.bin
 ```
- Options:
 - `--model`: one of `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`
 - `--output`: required — path to write the Mila weight file
 - `--dtype`: (optional) `float32` (default), `float16`, or `bfloat16`

- Other scripts
- See individual script docstrings / top-of-file usage comments for details and examples.

# Notes

- Keep the virtual environment activated while running scripts so the installed dependencies are used.
- If you need GPU-enabled PyTorch, install a CUDA-compatible wheel instead of the CPU wheel step above. Adjust instructions according to your platform and CUDA version.