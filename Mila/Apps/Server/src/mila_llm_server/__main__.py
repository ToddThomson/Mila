"""Makes `python -m mila_llm_server` equivalent to the mila-server console script."""
import sys

from mila_llm_server.cli import main

sys.exit(main())
