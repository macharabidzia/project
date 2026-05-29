from __future__ import annotations

from voice_pipeline.gpu.vllm_worker.engine import VLLMEngine, VLLMEngineConfig, build_prompt_cache_key


def test_prompt_cache_key_is_stable_for_same_scaffold() -> None:
    left = build_prompt_cache_key(
        system_prompt="You are voice runtime.",
        context_prefix="stable session summary",
        stable_prefix="committed user text",
    )
    right = build_prompt_cache_key(
        system_prompt="You are voice runtime.",
        context_prefix="stable session summary",
        stable_prefix="committed user text",
    )
    assert left == right
    assert "|" in left


def test_prefix_cache_ready_after_prewarm_without_extra_state() -> None:
    engine = VLLMEngine(
        "D:/models/vllm/Qwen3-8B",
        config=VLLMEngineConfig(model_name="D:/models/vllm/Qwen3-8B", model_path="D:/models/vllm/Qwen3-8B"),
    )
    engine._warm = True
    engine.prewarm_prefix_cache("voice_system_prefix", "stable_session_scaffold")

    assert engine.prefix_cache_ready is True
    stats = engine.cache_stats()
    assert stats.hits == 0
    assert stats.misses == 0
