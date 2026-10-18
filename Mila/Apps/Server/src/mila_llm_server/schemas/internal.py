"""
Canonical internal types that flow between protocol adapters and the route
factory. The worker speaks these types exclusively; adapters translate to and
from them.
"""
from dataclasses import dataclass, field


@dataclass
class Sampling:
    """
    The sampling a request asked for. Each knob left None is the model's publisher's setting for `thinking` --
    whether the prompt was rendered with thinking on -- and None there is the template's default
    (ChatConfiguration.md section 5). The binding resolves both. A seed reseeds the sampler before the reply;
    None continues from the previous one.
    """

    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    presence_penalty: float | None = None
    thinking: bool | None = None
    seed: int | None = None


@dataclass
class InferenceRequest:
    prompt_ids: list[int]
    # None when the request names no length: it may reply until the context is full.
    max_new_tokens: int | None
    stream: bool
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    # OpenAI's stop sequences. The reply is cut at the first one and the sequence itself is
    # not returned, streamed or buffered.
    stop: list[str] = field(default_factory=list)

    @property
    def sampling(self) -> Sampling:
        return Sampling(self.temperature, self.top_k, self.top_p, self.presence_penalty, seed=self.seed)


@dataclass
class InferenceResponse:
    text: str
    # Why the reply ended, in the routes' own terms (protocols.utils END_TURN and its siblings);
    # each adapter spells it for its wire. stop_sequence names the sequence that cut it, if one did.
    finish_reason: str = "end_turn"
    stop_sequence: str | None = None
    prompt_token_count: int = 0
    completion_token_count: int = 0
