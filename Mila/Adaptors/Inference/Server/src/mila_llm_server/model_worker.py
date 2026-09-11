"""
Owns the loaded model and its BpeTokenizer and serializes inference requests
through a single worker thread, honoring the model's non-thread-safe contract.
All public methods are async-safe and may be called from any asyncio task.

The model is named, not pathed: MILA_MODEL is a name in the local Mila store,
which Chat shares. Load never downloads -- see _load().
"""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import mila
from mila_llm_server.config import settings, loaded, ModelFamily
from mila_llm_server import gemma_bridge
from mila_llm_server.gemma_bridge import strip_control_tokens as _strip_gemma_control_tokens

# The checkpoint's own tokens, reported by the runtime rather than written down again.
TOOL_CALL_CLOSE = gemma_bridge.TOKENS["tool_call_close"]
TOOL_RESPONSE_OPEN = gemma_bridge.TOKENS["tool_response_open"]
CHANNEL_OPEN = gemma_bridge.TOKENS["reasoning_open"]

# Degeneration backstop. A bad Gemma sample can fail to emit a tool call or stop
# token and instead spam empty reasoning channels / repeat one token for the whole
# budget (observed: the literal "thought" label perseverated ~1000 tokens). These
# caps only fire on genuine runaway (hundreds) -- set well clear of any legitimate
# response, since a false trip cuts a real answer short (a too-tight channel cap
# once killed tool calls mid-reasoning).
_MAX_REASONING_CHANNELS = 24
_MAX_TOKEN_REPEATS = 48

_log = logging.getLogger(__name__)

#: The session class that loads each family's weights. A mapping rather than a chain of
#: conditionals, because the chain's else branch was a silent default: adding a third family
#: to ModelFamily routed its weights into LlamaModel rather than failing.
SESSION_FOR = {
    ModelFamily.gemma: "GemmaModel",
    ModelFamily.llama: "LlamaModel",
    ModelFamily.qwen: "QwenModel",
}


def _stop_markers_for(family: ModelFamily) -> tuple[str, ...]:
    """
    The decoded text that ends a turn early, in the loaded family's own grammar.

    A closing tool-call marker is a protocol boundary: left running, the model fabricates the
    tool result itself. The opening tool-response marker is the engine's turn to speak and never
    the model's, so it backstops a call the model failed to close.

    Qwen's come from the runtime rather than from constants here -- the grammar is the library's
    and asking it is what keeps this from becoming a second place the tokens are written down.
    """
    if family == ModelFamily.gemma:
        return (TOOL_CALL_CLOSE, TOOL_RESPONSE_OPEN)

    if family == ModelFamily.qwen:
        tokens = mila.qwen_protocol_tokens()
        return (tokens["tool_call_close"], tokens["tool_response_open"])

    return ()


def _family_of(record: "mila.StoredModel") -> ModelFamily:
    """
    The architecture the record declares, as the family the protocol layer branches on.

    An architecture MIS cannot serve is refused at load rather than at the first request:
    the store's vocabulary is wider than the binding's (gpt2 has a record shape and no
    session), and a server that started on one would fail every request instead.
    """
    try:
        return ModelFamily(record.architecture)
    except ValueError:
        raise RuntimeError(
            f"'{record.name}' has architecture '{record.architecture}', which MIS does not "
            f"serve. Supported: {', '.join(family.value for family in ModelFamily)}."
        ) from None


class ModelWorker:
    """
    Wraps the loaded model and its BpeTokenizer behind a single-thread executor so
    that FastAPI's async handlers can await inference without blocking the event loop
    or violating the model's single-threaded contract.
    """

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mila-worker")
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tokenizer: mila.BpeTokenizer | None = None
        self._model: "mila.LlamaModel | mila.GemmaModel | mila.QwenModel | None" = None

        # Decoded text that must halt generation, from the loaded family's own grammar. Empty
        # for a family with no tool spans, and filled by _load() -- never written down here,
        # since the runtime reports them.
        self._stop_markers: tuple[str, ...] = ()

    async def startup(self) -> None:
        self._loop = asyncio.get_running_loop()
        await self._loop.run_in_executor(self._executor, self._load)

    async def shutdown(self) -> None:
        self._executor.shutdown(wait=True)

    def _load(self) -> None:
        """
        Resolve the configured name against the local store and load what it names.

        MIS never pulls. A model arrives in the store through a deliberate act -- Chat's
        /install, or ExportArtifact --install -- and a server that downloaded 6 GB because
        a name was misspelled would be a worse failure than refusing to start.
        """
        store = mila.ModelStore()
        record = store.locate(settings.model)

        if record is None:
            installed = ", ".join(model.name for model in store.list())

            raise RuntimeError(
                f"No model named '{settings.model}' is installed in the Mila store "
                f"({store.root}). Installed: {installed or 'nothing'}.\n"
                "MIS loads only what is already installed. Install one with the chat "
                "harness (/install <name>), or with ExportArtifact --install from a "
                "source build, then start again. In a container, chat is a second "
                "entrypoint of this same image and shares this store."
            )

        loaded.name = record.name
        loaded.variant = record.variant
        loaded.instruct = record.instruct
        loaded.family = _family_of(record)
        loaded.base_model = record.base_model
        loaded.license = record.license

        # A server presents its model in its log and in /v1/models, and nowhere else. Logged
        # before the weights load so the attribution survives a load that fails.
        _log.info(
            "Loading %s (base model %s, license %s)",
            record.name,
            record.base_model or "unknown",
            record.license or "unstated",
        )

        if loaded.attribution:
            _log.info("%s", loaded.attribution)

        # One call for either family: the record says which loader the weights need, so
        # there is no longer a tokenizer path to pair with a weights path by hand.
        self._tokenizer = mila.BpeTokenizer.from_store(record.name)

        self._stop_markers = _stop_markers_for(loaded.family)

        session = getattr(mila, SESSION_FOR[loaded.family])

        # from_store, not from_pretrained: every published model is already quantized,
        # and only the record knows to what.
        self._model = session.from_store(
            record.name,
            settings.context_length,
            settings.device_index,
        )

    @property
    def _is_gemma(self) -> bool:
        return loaded.family == ModelFamily.gemma

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
            # The binding streams tokens and returns why it stopped, so prompt + completion
            # is assembled here -- which is the shape this worker's callers slice.
            output = list(prompt_tokens)
            self._model.generate(
                prompt_tokens, output.append, max_new_tokens, temperature, top_k, top_p)

            return output

        return await loop.run_in_executor(self._executor, _run)

    async def generate_collect(
        self,
        prompt_tokens: list[int],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float = 1.0,
    ) -> tuple[str, int]:
        """
        Non-streaming generation that still honors the Gemma <tool_call|> stop and
        the degeneration backstop by driving the streaming primitive to completion
        and accumulating the RAW (unstripped) decode. The plain blocking generate()
        path has neither guard and worker.decode() strips the <|tool_call> markers,
        so a non-streaming tool-call turn cannot be detected there; this path is what
        the Anthropic/Responses tool flows need for a single-shot JSON response.
        Returns (raw_text, decoded_chunk_count). Callers that only want display text
        reduce via gemma_bridge.answer_text / strip_control_tokens.
        """
        parts: list[str] = []

        def on_text(text: str) -> None:
            parts.append(text)

        stop_ctrl = mila.StopController()
        await self.generate_streaming(
            prompt_tokens,
            on_text,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            stop_ctrl,
            strip_control_tokens=False,
        )
        return "".join(parts), len(parts)

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
        Runs the binding's generate() on the worker thread. Each token is decoded
        on the worker thread and delivered as a string via on_text, avoiding
        re-entrant calls back into the executor from the asyncio event loop.
        on_text is called from the worker thread; callers must use thread-safe
        delivery (e.g. loop.call_soon_threadsafe into an asyncio.Queue).
        Token IDs are buffered until they form a valid UTF-8 sequence to handle
        tokens that carry only a partial multi-byte code point.

        Gemma native tool calls: <|tool_call> ... <tool_call|> is a registered
        protocol boundary. Left running, the model fabricates the tool result
        itself, so generation is stopped the moment <tool_call|> is decoded --
        exactly mirroring the chat harness (Chat.ixx generateResponse). Per the
        Gemma 4 spec, <|tool_response> is ALSO a stop sequence: it is the engine's
        turn to supply the result, never the model's, so if the model runs past a
        malformed/unclosed call and starts emitting <|tool_response> we cut it off
        before it can hallucinate an execution result. When strip_control_tokens is
        False the raw decoded text (channel + tool-call markers intact) is delivered
        so the caller can parse the native grammar; the responses/tool path needs
        this, the plain-chat streaming path does not.
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

            # A closing tool-call marker ends the turn; an opening tool-response marker is the
            # engine's to write, never the model's, so it backstops an unclosed call. Both come
            # from the loaded family's grammar -- see _stop_markers_for.
            stop_now = any(marker in text for marker in self._stop_markers)

            if self._is_gemma and stop_ctrl is not None and _degenerating(text):
                stop_now = True

            if self._is_gemma and strip_control_tokens:
                text = _strip_gemma_control_tokens(text)

            on_text(text)

            if stop_now and stop_ctrl is not None:
                stop_ctrl.request_stop()

        def _run() -> None:
            self._model.generate(
                prompt_tokens, _on_token, max_new_tokens, temperature, top_k, top_p, stop_ctrl
            )

        await loop.run_in_executor(self._executor, _run)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    async def get_model_info(self) -> dict:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._model.get_config)


worker = ModelWorker()
