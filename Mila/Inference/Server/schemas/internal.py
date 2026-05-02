"""
Canonical internal types that flow between protocol adapters and the route
factory. The worker speaks these types exclusively; adapters translate to and
from them.
"""
from dataclasses import dataclass, field


@dataclass
class InferenceRequest:
    prompt_ids: list[int]
    max_new_tokens: int
    temperature: float
    top_k: int
    stream: bool


@dataclass
class InferenceResponse:
    text: str
    finish_reason: str = "stop"
    prompt_token_count: int = 0
    completion_token_count: int = 0
