from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Callable

from voice_pipeline.gpu.vllm_worker.engine import VLLMEngine


class VLLMTokenStreamer:
    def __init__(self, engine: VLLMEngine) -> None:
        self._engine = engine

    async def stream(
        self,
        prompt: str,
        *,
        cache_key: str = "",
        request_id: str = "",
        on_token: Callable[[str], None] | None = None,
    ) -> AsyncIterator[str]:
        async for token in self._engine.stream_tokens(prompt, cache_key=cache_key, request_id=request_id):
            if on_token is not None:
                on_token(token)
            yield token


__all__ = ["VLLMTokenStreamer"]




