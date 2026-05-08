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
from config import settings


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
        self._model: mila.LlamaModel | None = None

    async def startup(self) -> None:
        self._loop = asyncio.get_running_loop()
        await self._loop.run_in_executor(self._executor, self._load)

    async def shutdown(self) -> None:
        self._executor.shutdown(wait=True)

    def _load(self) -> None:
        self._tokenizer = mila.BpeTokenizer.load_llama32(settings.tokenizer_path)
        self._model = mila.LlamaModel.from_pretrained(
            settings.model_path,
            settings.context_length,
            settings.device_index,
            settings.strict_load,
        )

    # ------------------------------------------------------------------
    # Tokenizer helpers (cheap; run on the worker thread for consistency)
    # ------------------------------------------------------------------

    async def encode(self, text: str) -> list[int]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._tokenizer.encode, text)

    async def decode(self, ids: list[int]) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._tokenizer.decode, ids)

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
        stop_ctrl: mila.StopController,
    ) -> None:
        """
        Runs generate_streaming() on the worker thread. Each token is decoded
        on the worker thread and delivered as a string via on_text, avoiding
        re-entrant calls back into the executor from the asyncio event loop.
        on_text is called from the worker thread; callers must use
        thread-safe delivery (e.g. loop.call_soon_threadsafe into an asyncio.Queue).
        """
        loop = asyncio.get_running_loop()

        def _on_token(token_id: int) -> None:
            text = self._tokenizer.decode([token_id])
            on_text(text)

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
