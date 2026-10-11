"""
Mila Inference Server: the wire adaptor over the Mila runtime.

Deliberately empty of imports. Importing app here would pull FastAPI, uvicorn and the
mila binding into every `import mila_llm_server`, including the test suite, which needs
only the protocol modules and no GPU.
"""
