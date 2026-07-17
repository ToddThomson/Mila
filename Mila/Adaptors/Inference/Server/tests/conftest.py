"""
MIS is run in place (python main.py) rather than installed as a package, so its
modules are not importable from a tests/ subdirectory by default. Put the server
directory on sys.path so `import gemma_protocol` resolves the same module the
running server uses.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Importing an adapter pulls in config.settings, which requires the model paths and
# would otherwise only resolve from a .env in the working directory -- making the
# suite pass or fail depending on where pytest was invoked from. These are never
# loaded: no test touches a model. Real values in the environment still win.
os.environ.setdefault("MILA_MODEL_PATH", "unused-by-tests.bin")
os.environ.setdefault("MILA_TOKENIZER_PATH", "unused-by-tests.bin")
