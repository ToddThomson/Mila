"""
No sys.path manipulation: MIS installs as the mila_llm_server package (pip install -e .),
so the suite imports exactly the modules the running server does.

What the suite does rely on is config.loaded, which ModelWorker fills in at startup from
the store record -- no test starts a worker, so the adapters read its defaults (Gemma,
instruct). A test that needs another family sets config.loaded.family directly rather than
an environment variable: the family is a fact about the artifact, and there is no longer an
env var that can claim otherwise.
"""
