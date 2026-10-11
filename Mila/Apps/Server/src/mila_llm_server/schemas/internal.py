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
    top_p: float = 1.0
    # OpenAI's stop sequences. The reply is cut at the first one and the sequence itself is
    # not returned, streamed or buffered.
    stop: list[str] = field(default_factory=list)


@dataclass
class InferenceResponse:
    text: str
    # Why the reply ended, in the routes' own terms (protocols.utils END_TURN and its siblings);
    # each adapter spells it for its wire. stop_sequence names the sequence that cut it, if one did.
    finish_reason: str = "end_turn"
    stop_sequence: str | None = None
    prompt_token_count: int = 0
    completion_token_count: int = 0
