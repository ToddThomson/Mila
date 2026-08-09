"""
MIS is run in place (python main.py) rather than installed as a package, so its
modules are not importable from a tests/ subdirectory by default. Put the server
directory on sys.path so `import gemma_protocol` resolves the same module the
running server uses.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Every setting now has a default, so nothing has to be faked here for config to
# import. What the suite does rely on is config.loaded, which ModelWorker fills in at
# startup from the store record -- no test starts a worker, so the adapters read its
# defaults (Gemma, instruct). A test that needs another family sets config.loaded.family
# directly rather than an environment variable: the family is a fact about the artifact,
# and there is no longer an env var that can claim otherwise.
