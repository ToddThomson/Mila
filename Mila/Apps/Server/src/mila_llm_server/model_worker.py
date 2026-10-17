"""
Owns the loaded model, its BpeTokenizer and its chat protocol, and serializes inference requests
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
from mila_llm_server.config import settings, loaded
from mila_llm_server.schemas.internal import Sampling

# Degeneration backstop. A bad sample can fail to emit a call or stop token and instead
# spam empty reasoning channels / repeat one token for the whole budget (observed on Gemma:
# the literal "thought" label perseverated ~1000 tokens). These caps only fire on genuine
# runaway (hundreds) -- set well clear of any legitimate response, since a false trip cuts
# a real answer short (a too-tight channel cap once killed tool calls mid-reasoning).
_MAX_REASONING_CHANNELS = 24
_MAX_TOKEN_REPEATS = 48

_log = logging.getLogger(__name__)

#: The binding's session class for each loader a store record names (StoredModel.family). The one place the
#: server meets the model's family, and only to load it: the binding has no family-blind loader yet, which the
#: model handle supplies (ModelHandle.md 3.5).
SESSION_FOR = {
    "gemma": "GemmaModel",
    "llama": "LlamaModel",
    "qwen": "QwenModel",
}


def _session_of(record: "mila.StoredModel"):
    """
    The session that loads the record's weights. A model MIS cannot serve is refused at load rather than at
    the first request: the store holds models no session loads (gpt2 has a record and no session), and a server
    that started on one would fail every request instead.
    """
    session = SESSION_FOR.get(record.family)

    if session is None:
        raise RuntimeError(
            f"'{record.name}' has architecture '{record.architecture}', which MIS does not serve. "
            f"Supported: {', '.join(SESSION_FOR)}."
        )

    return getattr(mila, session)


def _device_record(device_index: int) -> dict | None:
    """The card the model loaded onto, as /v1/models reports it; None if the runtime does not list it."""
    for device in mila.cuda_devices():
        if device.index == device_index:
            major, minor = device.compute_capability

            return {
                "index": device.index,
                "name": device.name,
                "compute_capability": f"{major}.{minor}",
                "total_memory_bytes": device.total_memory_bytes,
                "pci_bus_id": f"{device.pci_domain:08x}:{device.pci_bus:02x}:{device.pci_device:02x}.0",
            }

    return None


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
        self._model = None

        #: The loaded model's chat template. None for a base model, which has none to converse through.
        self.protocol: mila.ChatProtocol | None = None

        # Decoded text that must halt generation, and the open of a reasoning channel the backstop counts,
        # both from the protocol's own markers.
        self._stop_markers: tuple[str, ...] = ()
        self._reasoning_open: str = ""

    async def startup(self) -> None:
        self._loop = asyncio.get_running_loop()
        await self._loop.run_in_executor(self._executor, self._load)

    async def shutdown(self) -> None:
        self._executor.shutdown(wait=True)

    def _load(self) -> None:
        """
        Resolve the configured name against the local store and load what it names.

        Loading never pulls. A model arrives in the store through a deliberate act --
        mila-server install, or Chat's /model install -- and a server that downloaded 6 GB because
        a name was misspelled would be a worse failure than refusing to start.
        """
        store = mila.ModelStore()
        record = store.locate(settings.model)

        if record is None:
            installed = ", ".join(model.name for model in store.list())

            raise RuntimeError(
                f"No model named '{settings.model}' is installed in the Mila store "
                f"({store.root}). Installed: {installed or 'nothing'}.\n"
                "MIS loads only what is already installed. Install it with "
                f"'mila-server install {settings.model}', then start again. "
                "'mila-server models --online' lists what is published."
            )

        session = _session_of(record)

        loaded.name = record.name
        loaded.variant = record.variant
        loaded.instruct = record.instruct
        loaded.tools = record.tools
        loaded.thinking = record.thinking
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

        # The template the model ships with, named by its manifest. Read before the weights load, so a model whose
        # manifest does not say which template it runs with is refused in seconds rather than after a load.
        if record.instruct:
            self.protocol = mila.ChatProtocol.for_model(record.name)
            markers = self.protocol.markers
            self._stop_markers = tuple(markers["stop_after"])
            self._reasoning_open = markers["reasoning_open"]

        self._tokenizer = mila.BpeTokenizer.from_store(record.name)

        # from_store, not load: every published model is already quantized,
        # and only the record knows to what.
        self._model = session.from_store(
            record.name,
            settings.context_length,
            settings.device_index,
            settings.kv_cache,
        )

        loaded.context_length = self._model.context_length
        _log.info("Context length %d (configured %s), %s KV cache",
                  loaded.context_length, settings.context_length, settings.kv_cache.upper())

        loaded.mila_version = mila.version()
        loaded.device = _device_record(settings.device_index)
        _log.info(
            "Mila %s on CUDA device %d: %s",
            loaded.mila_version,
            settings.device_index,
            loaded.device["name"] if loaded.device else "unknown",
        )

    # ------------------------------------------------------------------
    # Tokenizer helpers (cheap; run on the worker thread for consistency)
    # ------------------------------------------------------------------

    async def encode(self, text: str) -> list[int]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._tokenizer.encode, text)

    async def decode(self, ids: list[int]) -> str:
        """
        A whole reply's text, as the model wrote it. Its ids may stop inside a multi-byte character -- a reply
        cut at max_tokens -- and that tail becomes U+FFFD rather than an error, as HuggingFace decodes it.
        Streaming decodes strictly instead, holding such a tail until its next token arrives.
        """
        loop = asyncio.get_running_loop()

        return await loop.run_in_executor(self._executor, self._tokenizer.decode, ids, "replace")

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
        sampling: Sampling,
    ) -> tuple[list[int], str]:
        """Prompt and completion ids, and the binding's status: stop, length or context_limit."""
        loop = asyncio.get_running_loop()

        def _run() -> tuple[list[int], str]:
            # The binding streams tokens and returns why it stopped, so prompt + completion
            # is assembled here -- which is the shape this worker's callers slice.
            output = list(prompt_tokens)
            status = self._model.generate(prompt_tokens, output.append, max_new_tokens, **vars(sampling))

            return output, status

        return await loop.run_in_executor(self._executor, _run)

    # ------------------------------------------------------------------
    # Streaming generation
    # ------------------------------------------------------------------

    async def generate_streaming(
        self,
        prompt_tokens: list[int],
        on_text: Callable[[str], None],
        max_new_tokens: int,
        sampling: Sampling,
        stop_ctrl: mila.StopController | None = None,
        stop_at_markers: bool = True,
    ) -> str:
        """
        Runs the binding's generate() on the worker thread, delivering the decoded text as the model wrote it,
        markers and all: what it means is the protocol's to parse. on_text is called from the worker thread;
        callers must use thread-safe delivery (e.g. loop.call_soon_threadsafe into an asyncio.Queue). Token ids
        are buffered until they form a valid UTF-8 sequence, since a token can carry part of a code point.

        With stop_at_markers, generation stops at the protocol's stop markers: a closed call, since left
        running the model writes the result itself, and the open of a result, which is the engine's to write
        and backstops a call the model failed to close.

        Returns the binding's status. A stop this worker requested at a marker is the model's turn ending, and
        reads as `stop`; `cancelled` is left only for a stop the caller requested, whose reason the caller knows.
        """
        loop = asyncio.get_running_loop()
        token_buffer: list[int] = []
        guard = {"channels": 0, "last": None, "repeats": 0}
        ended_by_marker = False
        stop_markers = self._stop_markers if stop_at_markers else ()

        def _degenerating(raw_text: str) -> bool:
            # Runaway reasoning channels: no legitimate answer opens this many.
            if self._reasoning_open and self._reasoning_open in raw_text:
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

            stop_now = any(marker in text for marker in stop_markers)

            if stop_ctrl is not None and stop_at_markers and _degenerating(text):
                stop_now = True

            on_text(text)

            if stop_now and stop_ctrl is not None and not stop_ctrl.stop_requested:
                nonlocal ended_by_marker
                ended_by_marker = True
                stop_ctrl.request_stop()

        def _run() -> str:
            return self._model.generate(
                prompt_tokens, _on_token, max_new_tokens, **vars(sampling), stop_controller=stop_ctrl)

        status = await loop.run_in_executor(self._executor, _run)

        return "stop" if ended_by_marker else status

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    async def get_model_info(self) -> dict:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._model.get_config)


worker = ModelWorker()
