"""
Owns the LlamaModel and BpeTokenizer instances and serializes inference
requests through a single worker thread, honoring the model's non-thread-safe
contract. All public methods are async-safe and may be called from any
asyncio task.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import mila
from config import settings, ModelFamily
from gemma_protocol import (
    strip_control_tokens as _strip_gemma_control_tokens,
    TOOL_CALL_CLOSE,
    CHANNEL_OPEN,
)

# Degeneration backstop. A bad Gemma sample can fail to emit a tool call or stop
# token and instead spam empty reasoning channels / repeat one token for the whole
# budget (observed: the literal "thought" label perseverated ~1000 tokens). These
# caps only fire on genuine runaway (hundreds) -- set well clear of any legitimate
# response, since a false trip cuts a real answer short (a too-tight channel cap
# once killed tool calls mid-reasoning).
_MAX_REASONING_CHANNELS = 24
_MAX_TOKEN_REPEATS = 48


class ModelWorker:
    """
    Wraps LlamaModel and BpeTokenizer behind a single-thread executor so that
    FastAPI's async handlers can await inference without blocking the event loop
    or violating the model's single-threaded contract.
    """

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mila-worker")
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tokenizer: mila.BpeTokenizer | None = None
        self._model: "mila.LlamaModel | mila.GemmaModel | None" = None

    async def startup(self) -> None:
        self._loop = asyncio.get_running_loop()
        await self._loop.run_in_executor(self._executor, self._load)

    async def shutdown(self) -> None:
        self._executor.shutdown(wait=True)

    def _load(self) -> None:
        if settings.model_family == ModelFamily.gemma:
            self._tokenizer = mila.BpeTokenizer.load_gemma(settings.tokenizer_path)
            self._model = mila.GemmaModel.from_pretrained(
                settings.model_path,
                settings.context_length,
                settings.device_index,
            )
        else:
            self._tokenizer = mila.BpeTokenizer.load_llama32(settings.tokenizer_path)
            self._model = mila.LlamaModel.from_pretrained(
                settings.model_path,
                settings.context_length,
                settings.device_index,
            )

    @property
    def _is_gemma(self) -> bool:
        return settings.model_family == ModelFamily.gemma

    # ------------------------------------------------------------------
    # Tokenizer helpers (cheap; run on the worker thread for consistency)
    # ------------------------------------------------------------------

    async def encode(self, text: str) -> list[int]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._tokenizer.encode, text)

    async def decode(self, ids: list[int]) -> str:
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(self._executor, self._tokenizer.decode, ids)

        if self._is_gemma:
            text = _strip_gemma_control_tokens(text)

        return text

    @property
    def bos_token_id(self) -> int | None:
        return self._tokenizer.bos_token_id if self._tokenizer else None

    @property
    def eos_token_id(self) -> int | None:
        return self._tokenizer.eos_token_id if self._tokenizer else None

    # ------------------------------------------------------------------
    # Blocking generation (non-streaming)
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt_tokens: list[int],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float = 1.0,
    ) -> list[int]:
        loop = asyncio.get_running_loop()

        def _run() -> list[int]:
            return self._model.generate(prompt_tokens, max_new_tokens, temperature, top_k)

        return await loop.run_in_executor(self._executor, _run)

    # ------------------------------------------------------------------
    # Streaming generation
    # ------------------------------------------------------------------

    async def generate_streaming(
        self,
        prompt_tokens: list[int],
        on_text: Callable[[str], None],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float = 1.0,
        stop_ctrl: mila.StopController | None = None,
        strip_control_tokens: bool = True,
    ) -> None:
        """
        Runs generate_streaming() on the worker thread. Each token is decoded
        on the worker thread and delivered as a string via on_text, avoiding
        re-entrant calls back into the executor from the asyncio event loop.
        on_text is called from the worker thread; callers must use thread-safe
        delivery (e.g. loop.call_soon_threadsafe into an asyncio.Queue).
        Token IDs are buffered until they form a valid UTF-8 sequence to handle
        tokens that carry only a partial multi-byte code point.

        Gemma native tool calls: <|tool_call> ... <tool_call|> is a registered
        protocol boundary. Left running, the model fabricates the tool result
        itself, so generation is stopped the moment <tool_call|> is decoded --
        exactly mirroring the chat harness (Chat.ixx generateResponse). When
        strip_control_tokens is False the raw decoded text (channel + tool-call
        markers intact) is delivered so the caller can parse the native grammar;
        the responses/tool path needs this, the plain-chat streaming path does
        not.
        """
        loop = asyncio.get_running_loop()
        token_buffer: list[int] = []
        guard = {"channels": 0, "last": None, "repeats": 0}

        def _degenerating(raw_text: str) -> bool:
            # Runaway reasoning channels: no legitimate answer opens this many.
            if CHANNEL_OPEN in raw_text:
                guard["channels"] += 1
                if guard["channels"] > _MAX_REASONING_CHANNELS:
                    return True

            # A single token repeated far past any natural repetition.
            stripped = raw_text.strip()
            if stripped:
                if stripped == guard["last"]:
                    guard["repeats"] += 1
                    if guard["repeats"] > _MAX_TOKEN_REPEATS:
                        return True
                else:
                    guard["repeats"] = 0
                    guard["last"] = stripped

            return False

        def _on_token(token_id: int) -> None:
            token_buffer.append(token_id)
            try:
                text = self._tokenizer.decode(token_buffer)
            except UnicodeDecodeError:
                return

            token_buffer.clear()

            stop_now = self._is_gemma and TOOL_CALL_CLOSE in text
            if self._is_gemma and stop_ctrl is not None and _degenerating(text):
                stop_now = True

            if self._is_gemma and strip_control_tokens:
                text = _strip_gemma_control_tokens(text)

            on_text(text)

            if stop_now and stop_ctrl is not None:
                stop_ctrl.request_stop()

        def _run() -> None:
            self._model.generate_streaming(
                prompt_tokens, _on_token, max_new_tokens, temperature, top_k, stop_ctrl
            )

        await loop.run_in_executor(self._executor, _run)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    async def get_model_info(self) -> dict:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._model.get_config)


worker = ModelWorker()
