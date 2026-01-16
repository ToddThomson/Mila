cd Data\Scripts
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install CPU-only torch first to avoid large GPU wheels (optional but recommended)
pip install torch --index-url https://download.pytorch.org/whl/cpu
# Then install remaining requirements
pip install -r requirements.txt

