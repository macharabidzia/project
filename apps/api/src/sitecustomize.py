from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.metadata
import inspect
import multiprocessing
import os
import json
import site
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


_ORIGINAL_RGLOB = Path.rglob
_ORIGINAL_INSPECT_GETABSFILE = inspect.getabsfile
_ORIGINAL_INSPECT_GETSOURCEFILE = inspect.getsourcefile
_TEXT_ONLY_RUNTIME = os.getenv("VLLM_TEXT_ONLY_RUNTIME", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
if os.getenv("VLLM_TEXT_ONLY_RUNTIME") is None:
    _TEXT_ONLY_RUNTIME = True
if _TEXT_ONLY_RUNTIME and os.getenv("VLLM_SKIP_ENV_OVERRIDE_IMPORT") is None:
    os.environ["VLLM_SKIP_ENV_OVERRIDE_IMPORT"] = "1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _coerce_inspect_filename(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, os.PathLike):
        try:
            return os.fspath(value)
        except TypeError:
            return None
    return None


def _patch_inspect_filename_guards() -> None:
    def _fallback_filename(obj: Any) -> str:
        name = getattr(obj, "__name__", "") or type(obj).__name__
        return f"<voice_pipeline_unknown:{name}>"

    def _patched_getsourcefile(obj: Any) -> str | None:
        try:
            return _ORIGINAL_INSPECT_GETSOURCEFILE(obj)
        except AttributeError as exc:
            if "endswith" not in str(exc):
                raise
            try:
                filename = inspect.getfile(obj)
            except (OSError, TypeError):
                return _fallback_filename(obj)
            return _coerce_inspect_filename(filename) or _fallback_filename(obj)

    def _patched_getabsfile(obj: Any, _filename: Any = None) -> str:
        try:
            return _ORIGINAL_INSPECT_GETABSFILE(obj, _filename)
        except TypeError:
            filename = _coerce_inspect_filename(_filename)
            if filename is None:
                try:
                    filename = _patched_getsourcefile(obj)
                except (OSError, TypeError):
                    filename = None
            if filename is None:
                try:
                    filename = _coerce_inspect_filename(inspect.getfile(obj))
                except (OSError, TypeError):
                    filename = None
            if filename is None:
                filename = _fallback_filename(obj)
            return os.path.normcase(os.path.abspath(filename))

    inspect.getsourcefile = _patched_getsourcefile
    inspect.getabsfile = _patched_getabsfile


def _prepend_if_dir(path: Path) -> None:
    if not path.exists() or not path.is_dir():
        return
    resolved = str(path)
    if resolved in sys.path:
        return
    sys.path.insert(0, resolved)


def _prepend_cosyvoice_runtime_paths() -> None:
    configured_repo = os.getenv("COSYVOICE_REPO_DIR", "").strip()
    cosyvoice_root = (
        Path(configured_repo).expanduser().resolve()
        if configured_repo
        else (_repo_root() / ".models" / "CosyVoice-runtime")
    )
    _prepend_if_dir(cosyvoice_root)
    _prepend_if_dir(cosyvoice_root / "third_party" / "Matcha-TTS")


def _prepend_cuda_runtime_library_paths() -> None:
    resolved = os.getenv("VOICE_PIPELINE_RESOLVED_LD_LIBRARY_PATH", "").strip()
    if resolved:
        library_paths = [entry for entry in resolved.split(os.pathsep) if entry]
    else:
        library_paths = []
        for site_dir in site.getsitepackages():
            root = Path(site_dir)
            torch_lib = root / "torch" / "lib"
            if torch_lib.is_dir():
                library_paths.append(str(torch_lib))
            nvidia_root = root / "nvidia"
            if nvidia_root.is_dir():
                for lib_dir in sorted(nvidia_root.glob("*/lib")):
                    if lib_dir.is_dir():
                        library_paths.append(str(lib_dir))
    if not library_paths:
        return
    existing = [entry for entry in str(os.getenv("LD_LIBRARY_PATH", "")).split(os.pathsep) if entry]
    merged: list[str] = []
    seen: set[str] = set()
    for candidate in [*library_paths, *existing]:
        if candidate in seen:
            continue
        seen.add(candidate)
        merged.append(candidate)
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(merged)


def _append_runtime_sys_path_entries() -> None:
    raw = os.getenv("VOICE_PIPELINE_APPEND_SYS_PATH", "").strip()
    if not raw:
        return

    for entry in raw.split(os.pathsep):
        entry = entry.strip()
        if not entry or entry in sys.path:
            continue
        sys.path.append(entry)


def _prepend_runtime_sys_path_entries() -> None:
    raw = os.getenv("VOICE_PIPELINE_PREPEND_SYS_PATH", "").strip()
    if not raw:
        return

    entries = [entry.strip() for entry in raw.split(os.pathsep) if entry.strip()]
    for entry in reversed(entries):
        if entry in sys.path:
            continue
        sys.path.insert(0, entry)


def _should_skip_transformers_image_processor_scan(self: Path, pattern: str) -> bool:
    if os.getenv("TRANSFORMERS_SKIP_IMAGE_PROCESSOR_ALIAS_SCAN", "").strip() not in {"1", "true", "yes", "on"}:
        return False
    return (
        pattern == "image_processing_*.py"
        and self.name == "models"
        and self.parent.name == "transformers"
    )


def _patched_rglob(self: Path, pattern: str):
    if _should_skip_transformers_image_processor_scan(self, pattern):
        return iter(())
    return _ORIGINAL_RGLOB(self, pattern)


class _PatchedLoader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader, patcher: Callable[[Any], None]) -> None:
        self._wrapped = wrapped
        self._patcher = patcher

    def create_module(self, spec):
        if hasattr(self._wrapped, "create_module"):
            return self._wrapped.create_module(spec)  # type: ignore[misc]
        return None

    def exec_module(self, module) -> None:
        self._wrapped.exec_module(module)
        if getattr(module, "__voice_pipeline_patch_applied__", False):
            return
        self._patcher(module)
        setattr(module, "__voice_pipeline_patch_applied__", True)


class _PostImportPatchFinder(importlib.abc.MetaPathFinder):
    def __init__(self) -> None:
        self._patchers: dict[str, Callable[[Any], None]] = {}

    def register(self, module_name: str, patcher: Callable[[Any], None]) -> None:
        self._patchers[module_name] = patcher
        existing = sys.modules.get(module_name)
        if existing is not None and not getattr(existing, "__voice_pipeline_patch_applied__", False):
            patcher(existing)
            setattr(existing, "__voice_pipeline_patch_applied__", True)

    def find_spec(self, fullname: str, path, target=None):
        patcher = self._patchers.get(fullname)
        if patcher is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _PatchedLoader(spec.loader, patcher)
        return spec


_POST_IMPORT_PATCH_FINDER = _PostImportPatchFinder()


class _StubModuleLoader(importlib.abc.Loader):
    def __init__(self, module_name: str, attrs: dict[str, Any], *, is_package: bool = False) -> None:
        self._module_name = module_name
        self._attrs = dict(attrs)
        self._is_package = bool(is_package)

    def create_module(self, spec):
        module = types.ModuleType(spec.name)
        module.__file__ = f"<voice_pipeline_stub:{spec.name}>"
        module.__package__ = spec.name if self._is_package else spec.name.rpartition(".")[0]
        if self._is_package:
            module.__path__ = []  # type: ignore[attr-defined]
        return module

    def exec_module(self, module) -> None:
        module.__dict__.update(self._attrs)
        if self._is_package and "__path__" in self._attrs:
            module.__path__ = list(self._attrs["__path__"])  # type: ignore[attr-defined]


class _StubModuleFinder(importlib.abc.MetaPathFinder):
    def __init__(self) -> None:
        self._stubs: dict[str, tuple[dict[str, Any], bool]] = {}

    def register(self, module_name: str, attrs: dict[str, Any], *, is_package: bool = False) -> None:
        self._stubs[module_name] = (dict(attrs), bool(is_package))

    def find_spec(self, fullname: str, path, target=None):
        stub = self._stubs.get(fullname)
        if stub is None:
            return None
        attrs, is_package = stub
        loader = _StubModuleLoader(fullname, attrs, is_package=is_package)
        return importlib.machinery.ModuleSpec(fullname, loader, is_package=is_package)


_STUB_MODULE_FINDER = _StubModuleFinder()


def _ensure_post_import_patch_finder() -> None:
    if _POST_IMPORT_PATCH_FINDER in sys.meta_path:
        return
    sys.meta_path.insert(0, _POST_IMPORT_PATCH_FINDER)


def _ensure_stub_module_finder() -> None:
    if _STUB_MODULE_FINDER in sys.meta_path:
        return
    sys.meta_path.insert(0, _STUB_MODULE_FINDER)


def _install_stub_module(module_name: str, attrs: dict[str, Any], *, is_package: bool = False) -> None:
    _ensure_stub_module_finder()
    _STUB_MODULE_FINDER.register(module_name, attrs, is_package=is_package)
    module = types.ModuleType(module_name)
    module.__dict__.update(attrs)
    module.__file__ = f"<voice_pipeline_stub:{module_name}>"
    module.__package__ = module_name if is_package else module_name.rpartition(".")[0]
    if is_package:
        module.__path__ = []  # type: ignore[attr-defined]
        if "__path__" in attrs:
            module.__path__ = list(attrs["__path__"])  # type: ignore[attr-defined]
    sys.modules[module_name] = module


def _install_post_import_patch(module_name: str, patcher: Callable[[Any], None]) -> None:
    _ensure_post_import_patch_finder()
    _POST_IMPORT_PATCH_FINDER.register(module_name, patcher)


def _patch_importlib_metadata() -> None:
    if not _TEXT_ONLY_RUNTIME:
        return
    original_entry_points = importlib.metadata.entry_points

    def _patched_entry_points(*args, **kwargs):
        group = kwargs.get("group")
        if group in {"vllm.logits_processors", "pydantic"}:
            return ()
        return original_entry_points(*args, **kwargs)

    importlib.metadata.entry_points = _patched_entry_points


def _install_deep_gemm_stub() -> None:
    if not _TEXT_ONLY_RUNTIME or "vllm.utils.deep_gemm" in sys.modules:
        return

    module = types.ModuleType("vllm.utils.deep_gemm")
    module.__dict__["__all__"] = [
        "calc_diff",
        "DeepGemmQuantScaleFMT",
        "fp8_gemm_nt",
        "fp8_einsum",
        "m_grouped_fp8_gemm_nt_contiguous",
        "m_grouped_fp8_fp4_gemm_nt_contiguous",
        "fp8_m_grouped_gemm_nt_masked",
        "fp8_fp4_mqa_logits",
        "fp8_fp4_paged_mqa_logits",
        "get_paged_mqa_logits_metadata",
        "per_block_cast_to_fp8",
        "is_deep_gemm_e8m0_used",
        "is_deep_gemm_supported",
        "get_num_sms",
        "set_num_sms",
        "should_use_deepgemm_for_fp8_linear",
        "get_col_major_tma_aligned_tensor",
        "get_tma_aligned_size",
        "get_mk_alignment_for_contiguous_layout",
        "transform_sf_into_required_layout",
        "should_auto_disable_deep_gemm",
    ]

    class DeepGemmQuantScaleFMT:
        FLOAT32 = 0
        FLOAT32_CEIL_UE8M0 = 1
        UE8M0 = 2

    def _disabled(*args, **kwargs):
        raise RuntimeError("deep_gemm disabled in VLLM_TEXT_ONLY_RUNTIME")

    def _false(*args, **kwargs):
        return False

    def _zero(*args, **kwargs):
        return 0

    def _identity_cast(x, *args, **kwargs):
        try:
            import torch

            scales = torch.ones((1, 1), dtype=torch.float32, device=x.device)
            return x, scales
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("per_block_cast_to_fp8 fallback unavailable") from exc

    module.DeepGemmQuantScaleFMT = DeepGemmQuantScaleFMT
    module.calc_diff = _disabled
    module.fp8_gemm_nt = _disabled
    module.fp8_einsum = _disabled
    module.m_grouped_fp8_gemm_nt_contiguous = _disabled
    module.m_grouped_fp8_fp4_gemm_nt_contiguous = _disabled
    module.fp8_m_grouped_gemm_nt_masked = _disabled
    module.fp8_fp4_mqa_logits = _disabled
    module.fp8_fp4_paged_mqa_logits = _disabled
    module.get_paged_mqa_logits_metadata = _disabled
    module.per_block_cast_to_fp8 = _identity_cast
    module.is_deep_gemm_e8m0_used = _false
    module.is_deep_gemm_supported = _false
    module.get_num_sms = _zero
    module.set_num_sms = lambda *_args, **_kwargs: None
    module.should_use_deepgemm_for_fp8_linear = _false
    module.get_col_major_tma_aligned_tensor = _disabled
    module.get_tma_aligned_size = lambda size, *args, **kwargs: size
    module.get_mk_alignment_for_contiguous_layout = lambda *_args, **_kwargs: 1
    module.transform_sf_into_required_layout = lambda x, *args, **kwargs: x
    module.should_auto_disable_deep_gemm = _false
    module.__getattr__ = lambda _name: _disabled
    sys.modules[module.__name__] = module


def _install_torch_dynamo_stub() -> None:
    if not _TEXT_ONLY_RUNTIME:
        return

    def _identity_decorator(fn=None, *args, **kwargs):
        if fn is None:
            def _wrap(inner):
                return inner
            return _wrap
        return fn

    attrs = {
        "disable": _identity_decorator,
        "optimize": _identity_decorator,
        "optimize_assert": _identity_decorator,
        "run": _identity_decorator,
        "graph_break": (lambda *args, **kwargs: None),
        "reset": (lambda *args, **kwargs: None),
        "is_compiling": (lambda *args, **kwargs: False),
        "allow_in_graph": _identity_decorator,
        "disallow_in_graph": _identity_decorator,
        "assume_constant_result": _identity_decorator,
        "trace_rules": type(
            "TraceRules",
            (),
            {"clear_lru_cache": staticmethod(lambda: None)},
        )(),
        "mark_dynamic": (lambda *args, **kwargs: None),
        "mark_static": (lambda *args, **kwargs: None),
        "__all__": [
        "disable",
        "optimize",
        "optimize_assert",
        "run",
        "graph_break",
        "reset",
        "is_compiling",
        "allow_in_graph",
        "disallow_in_graph",
        "assume_constant_result",
        "trace_rules",
        "mark_dynamic",
        "mark_static",
        ],
    }
    _install_stub_module("torch._dynamo", attrs, is_package=True)
    _install_stub_module(
        "torch._dynamo.utils",
        {
            "counters": {},
            "warn_once_cache": {},
            "warn_once": (lambda *args, **kwargs: None),
            "dynamo_timed": _identity_decorator,
            "get_metrics_context": (lambda *args, **kwargs: None),
            "defake": (lambda x, *args, **kwargs: x),
            "flatten_graph_inputs": (lambda *args, **kwargs: []),
            "deepcopy_to_fake_tensor": (lambda x, *args, **kwargs: x),
            "detect_fake_mode": (lambda *args, **kwargs: None),
            "to_fake_tensor": (lambda x, *args, **kwargs: x),
            "_disable_saved_tensors_hooks_during_tracing": (
                lambda *args, **kwargs: _null_context()
            ),
            "_disable_side_effect_safety_checks_for_current_subtracer": (
                lambda *args, **kwargs: None
            ),
            "object_has_getattribute": (lambda *args, **kwargs: False),
            "get_fake_value": (lambda *args, **kwargs: None),
            "get_static_address_type": (lambda *args, **kwargs: None),
            "invalid_removeable_handle": object(),
            "dict_keys": dict().keys().__class__,
            "logging": __import__("logging"),
            "CompileEventLogger": object,
            "ReinplaceCounters": type(
                "ReinplaceCounters",
                (),
                {"clear": staticmethod(lambda: None), "log": staticmethod(lambda: None)},
            ),
            "is_safe_constant": (lambda *args, **kwargs: False),
            "get_optimize_ddp_mode": (lambda *args, **kwargs: None),
            "is_compile_supported": (lambda *args, **kwargs: False),
        },
    )
    _install_stub_module(
        "torch._dynamo.aot_compile_types",
        {
            "BundledAOTAutogradSerializableCallable": type(
                "BundledAOTAutogradSerializableCallable",
                (),
                {
                    "__init__": lambda self, fn, *args, **kwargs: setattr(self, "inner_fn", fn),
                    "__call__": lambda self, *args, **kwargs: self.inner_fn(*args, **kwargs),
                    "serialize_compile_artifacts": staticmethod(
                        lambda payload, *args, **kwargs: payload
                    ),
                    "deserialize_compile_artifacts": staticmethod(
                        lambda payload, *args, **kwargs: payload
                    ),
                },
            ),
        },
    )


def _null_context():
    from contextlib import nullcontext

    return nullcontext()


def _install_pyworld_stub() -> None:
    if not _TEXT_ONLY_RUNTIME:
        return
    if "pyworld" in sys.modules:
        return

    def _unsupported(*args, **kwargs):
        raise RuntimeError("pyworld disabled in VLLM_TEXT_ONLY_RUNTIME")

    _install_stub_module(
        "pyworld",
        {
            "harvest": _unsupported,
            "dio": _unsupported,
            "stonemask": _unsupported,
            "__all__": ["harvest", "dio", "stonemask"],
        },
    )


def _install_whisper_stub() -> None:
    if not _TEXT_ONLY_RUNTIME:
        return
    if "whisper" in sys.modules:
        return

    def _log_mel_spectrogram(audio: Any, *, n_mels: int = 80, padding: int = 0) -> Any:
        import torch
        import torchaudio

        waveform = torch.as_tensor(audio, dtype=torch.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim > 2:
            waveform = waveform.reshape(1, -1)
        if padding > 0:
            waveform = torch.nn.functional.pad(waveform, (0, int(padding)))

        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            win_length=400,
            hop_length=160,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            mel_scale="slaney",
            n_mels=int(n_mels),
        )(waveform)
        mel = torch.clamp(mel, min=1e-10).log10()
        mel = torch.maximum(mel, mel.max() - 8.0)
        mel = (mel + 4.0) / 4.0
        return mel

    class _StubWhisperTokenizer:
        def __init__(
            self,
            encoding: Any,
            *,
            num_languages: int = 99,
            language: str | None = None,
            task: str | None = None,
        ) -> None:
            self.encoding = encoding
            self.num_languages = num_languages
            self.language = language
            self.task = task

        def encode(self, text: str, *, allowed_special: Any = "all") -> list[int]:
            return list(self.encoding.encode(text, allowed_special=allowed_special))

        def decode(self, tokens: list[int] | tuple[int, ...]) -> str:
            return str(self.encoding.decode(list(tokens)))

    _install_stub_module(
        "whisper",
        {
            "log_mel_spectrogram": _log_mel_spectrogram,
            "tokenizer": None,
            "__all__": ["log_mel_spectrogram", "tokenizer"],
        },
        is_package=True,
    )
    _install_stub_module(
        "whisper.tokenizer",
        {
            "Tokenizer": _StubWhisperTokenizer,
            "__all__": ["Tokenizer"],
        },
    )
    whisper_pkg = sys.modules.get("whisper")
    whisper_tokenizer_mod = sys.modules.get("whisper.tokenizer")
    if whisper_pkg is not None and whisper_tokenizer_mod is not None:
        setattr(whisper_pkg, "tokenizer", whisper_tokenizer_mod)


def _install_cosyvoice_dataset_processor_stub() -> None:
    if not _TEXT_ONLY_RUNTIME:
        return
    if "cosyvoice.dataset.processor" in sys.modules:
        return

    def _identity_iter(data, *args, **kwargs):
        return data

    def _tokenize(data, *args, **kwargs):
        return data

    def _batch(data, *args, **kwargs):
        return data

    attrs = {
        "parquet_opener": _identity_iter,
        "filter": _identity_iter,
        "resample": _identity_iter,
        "truncate": _identity_iter,
        "compute_fbank": _identity_iter,
        "compute_whisper_fbank": _identity_iter,
        "compute_f0": _identity_iter,
        "parse_embedding": _identity_iter,
        "tokenize": _tokenize,
        "shuffle": _identity_iter,
        "sort": _identity_iter,
        "static_batch": _batch,
        "dynamic_batch": _batch,
        "batch": _batch,
        "padding": _identity_iter,
        "__all__": [
            "parquet_opener",
            "filter",
            "resample",
            "truncate",
            "compute_fbank",
            "compute_whisper_fbank",
            "compute_f0",
            "parse_embedding",
            "tokenize",
            "shuffle",
            "sort",
            "static_batch",
            "dynamic_batch",
            "batch",
            "padding",
        ],
    }
    _install_stub_module("cosyvoice.dataset", {}, is_package=True)
    _install_stub_module("cosyvoice.dataset.processor", attrs)
    dataset_pkg = sys.modules.get("cosyvoice.dataset")
    processor_mod = sys.modules.get("cosyvoice.dataset.processor")
    if dataset_pkg is not None and processor_mod is not None:
        setattr(dataset_pkg, "processor", processor_mod)

    def _attach_dataset_pkg(cosyvoice_module: Any) -> None:
        dataset_module = sys.modules.get("cosyvoice.dataset")
        if dataset_module is not None:
            setattr(cosyvoice_module, "dataset", dataset_module)

    _install_post_import_patch("cosyvoice", _attach_dataset_pkg)


def _install_vllm_multimodal_stubs() -> None:
    if not _TEXT_ONLY_RUNTIME:
        return

    class _StubMultiModalRegistry:
        def supports_multimodal_inputs(self, *args, **kwargs) -> bool:
            return False

        def create_processor(self, *args, **kwargs):
            raise RuntimeError("multimodal disabled in VLLM_TEXT_ONLY_RUNTIME")

        def create_input_mapper(self, *args, **kwargs):
            raise RuntimeError("multimodal disabled in VLLM_TEXT_ONLY_RUNTIME")

        def init_mm_limits_per_prompt(self, *args, **kwargs):
            return {}

        def get_mm_limits_per_prompt(self, *args, **kwargs):
            return {}

    registry = _StubMultiModalRegistry()

    class _PlaceholderRange:
        def __init__(self, offset: int = 0, length: int = 0, is_embed: Any = None, **kwargs) -> None:
            self.offset = offset
            self.length = length
            self.is_embed = is_embed
            for key, value in kwargs.items():
                setattr(self, key, value)

    class _MultiModalSharedField:
        def __init__(self, batch_size: int = 1, **kwargs) -> None:
            self.batch_size = batch_size
            for key, value in kwargs.items():
                setattr(self, key, value)

    class _MultiModalFieldElem:
        def __init__(self, data: Any = None, field: Any = None, **kwargs) -> None:
            self.data = data
            self.field = field
            for key, value in kwargs.items():
                setattr(self, key, value)

    class _MultiModalKwargsItem(dict):
        pass

    class _MultiModalHasher:
        @staticmethod
        def hash_kwargs(**kwargs) -> str:
            return "voice-pipeline-mm-disabled"
    _install_stub_module(
        "vllm.multimodal",
        {
            "MultiModalRegistry": _StubMultiModalRegistry,
            "MULTIMODAL_REGISTRY": registry,
            "BatchedTensorInputs": dict,
            "MultiModalKwargsItems": tuple,
            "NestedTensors": tuple,
            "MultiModalHasher": _MultiModalHasher,
            "__all__": [
                "BatchedTensorInputs",
                "MultiModalHasher",
                "MultiModalKwargsItems",
                "NestedTensors",
                "MULTIMODAL_REGISTRY",
                "MultiModalRegistry",
            ],
        },
        is_package=True,
    )
    _install_stub_module(
        "vllm.multimodal.registry",
        {
            "MultiModalRegistry": _StubMultiModalRegistry,
            "MULTIMODAL_REGISTRY": registry,
        },
    )
    _install_stub_module(
        "vllm.multimodal.inputs",
        {
            "MultiModalFeatureSpec": Any,
            "MultiModalKwargsItem": _MultiModalKwargsItem,
            "MultiModalFieldElem": _MultiModalFieldElem,
            "MultiModalSharedField": _MultiModalSharedField,
            "MultiModalBatchedField": dict,
            "MultiModalFlatField": dict,
            "PlaceholderRange": _PlaceholderRange,
            "VisionChunk": dict,
            "VisionChunkImage": dict,
            "VisionChunkVideo": dict,
            "BatchedTensorInputs": dict,
            "MultiModalKwargsItems": tuple,
            "NestedTensors": tuple,
            "__all__": [
                "MultiModalFeatureSpec",
                "MultiModalKwargsItem",
                "MultiModalFieldElem",
                "MultiModalSharedField",
                "MultiModalBatchedField",
                "MultiModalFlatField",
                "PlaceholderRange",
                "VisionChunk",
                "VisionChunkImage",
                "VisionChunkVideo",
                "BatchedTensorInputs",
                "MultiModalKwargsItems",
                "NestedTensors",
            ],
        },
    )
    _install_stub_module(
        "vllm.multimodal.encoder_budget",
        {
            "MultiModalBudget": type(
                "MultiModalBudget",
                (),
                {
                    "__init__": lambda self, *args, **kwargs: None,
                    "get_modality_with_max_tokens": lambda self: "",
                    "get_encoder_budget": lambda self: 0,
                    "reset_cache": lambda self: None,
                    "mm_limits": {},
                    "mm_max_toks_per_item": {},
                    "mm_max_items_per_prompt": {},
                    "mm_max_items_per_batch": {},
                    "encoder_compute_budget": 0,
                    "encoder_cache_size": 0,
                },
            ),
            "__all__": ["MultiModalBudget"],
        },
    )
    _install_stub_module(
        "vllm.multimodal.utils",
        {
            "group_and_batch_mm_kwargs": (
                lambda mm_kwargs, *args, **kwargs: ()
                if not mm_kwargs
                else (item for item in ())
            ),
            "argsort_mm_positions": (lambda mm_positions, *args, **kwargs: list(range(len(mm_positions or [])))),
            "__all__": ["group_and_batch_mm_kwargs", "argsort_mm_positions"],
        },
    )
    class _StubMediaConnector:
        def __init__(self, *args, **kwargs) -> None:
            pass

    class _StubMediaConnectorRegistry:
        def load(self, *args, **kwargs):
            return _StubMediaConnector()

    _install_stub_module(
        "vllm.multimodal.media",
        {
            "MediaConnector": _StubMediaConnector,
            "MEDIA_CONNECTOR_REGISTRY": _StubMediaConnectorRegistry(),
            "__all__": ["MediaConnector", "MEDIA_CONNECTOR_REGISTRY"],
        },
        is_package=True,
    )
    _install_stub_module(
        "vllm.multimodal.media.connector",
        {
            "MediaConnector": _StubMediaConnector,
            "MEDIA_CONNECTOR_REGISTRY": _StubMediaConnectorRegistry(),
            "merge_media_io_kwargs": (lambda *args, **kwargs: {}),
            "__all__": [
                "MediaConnector",
                "MEDIA_CONNECTOR_REGISTRY",
                "merge_media_io_kwargs",
            ],
        },
    )
    _install_stub_module(
        "vllm.multimodal.hasher",
        {
            "MultiModalHasher": _MultiModalHasher,
            "__all__": ["MultiModalHasher"],
        },
    )


def _install_vllm_multimodal_processing_stubs() -> None:
    if not _TEXT_ONLY_RUNTIME:
        return

    class _BaseProcessingInfo:
        pass

    class _InputProcessingContext:
        pass

    class _TimingContext:
        def __init__(self, *args, **kwargs):
            self.enabled = bool(kwargs.get("enabled", True))
            self.stage_secs = {}

    class _BaseDummyInputsBuilder:
        pass

    class _ProcessorInputs(dict):
        pass

    class _BaseMultiModalProcessor:
        pass

    class _EncDecMultiModalProcessor(_BaseMultiModalProcessor):
        pass

    class _PromptIndexTargets:
        pass

    class _PromptUpdateDetails:
        pass

    class _PromptInsertion:
        pass

    class _PromptReplacement:
        def __init__(self, modality: str = "", target: Any = None, replacement: Any = None, **kwargs) -> None:
            self.modality = modality
            self.target = target
            self.replacement = replacement
            for key, value in kwargs.items():
                setattr(self, key, value)

        def resolve(self, item_idx: int = 0):
            return type(
                "ResolvedPromptUpdate",
                (),
                {
                    "modality": self.modality,
                    "target": self.target,
                    "replacement": self.replacement,
                    "item_idx": item_idx,
                },
            )()

    class _PromptUpdate:
        pass

    def _apply_token_matches(token_ids, mm_prompt_updates, tokenizer=None):
        return list(token_ids), mm_prompt_updates

    def _find_mm_placeholders(prompt, mm_prompt_updates, tokenizer=None):
        return {}

    common_attrs = {
        "BaseProcessingInfo": _BaseProcessingInfo,
        "InputProcessingContext": _InputProcessingContext,
        "TimingContext": _TimingContext,
        "BaseDummyInputsBuilder": _BaseDummyInputsBuilder,
        "ProcessorInputs": _ProcessorInputs,
        "BaseMultiModalProcessor": _BaseMultiModalProcessor,
        "EncDecMultiModalProcessor": _EncDecMultiModalProcessor,
        "PromptIndexTargets": _PromptIndexTargets,
        "PromptUpdateDetails": _PromptUpdateDetails,
        "PromptInsertion": _PromptInsertion,
        "PromptReplacement": _PromptReplacement,
        "PromptUpdate": _PromptUpdate,
    }
    _install_stub_module(
        "vllm.multimodal.processing",
        {
            **common_attrs,
            "__all__": [
                "BaseProcessingInfo",
                "InputProcessingContext",
                "TimingContext",
                "BaseDummyInputsBuilder",
                "ProcessorInputs",
                "BaseMultiModalProcessor",
                "EncDecMultiModalProcessor",
                "PromptUpdate",
                "PromptIndexTargets",
                "PromptUpdateDetails",
                "PromptInsertion",
                "PromptReplacement",
            ],
        },
        is_package=True,
    )
    _install_stub_module(
        "vllm.multimodal.processing.context",
        {
            "BaseProcessingInfo": _BaseProcessingInfo,
            "InputProcessingContext": _InputProcessingContext,
            "TimingContext": _TimingContext,
        },
    )
    _install_stub_module(
        "vllm.multimodal.processing.processor",
        {
            "BaseMultiModalProcessor": _BaseMultiModalProcessor,
            "EncDecMultiModalProcessor": _EncDecMultiModalProcessor,
            "PromptIndexTargets": _PromptIndexTargets,
            "PromptUpdateDetails": _PromptUpdateDetails,
            "PromptInsertion": _PromptInsertion,
            "PromptReplacement": _PromptReplacement,
            "PromptUpdate": _PromptUpdate,
            "apply_token_matches": _apply_token_matches,
            "find_mm_placeholders": _find_mm_placeholders,
        },
    )
    _install_stub_module(
        "vllm.multimodal.processing.dummy_inputs",
        {"BaseDummyInputsBuilder": _BaseDummyInputsBuilder},
    )
    _install_stub_module(
        "vllm.multimodal.processing.inputs",
        {"ProcessorInputs": _ProcessorInputs},
    )
    _install_stub_module("vllm.multimodal.audio", {}, is_package=False)
    _install_stub_module("vllm.multimodal.parse", {}, is_package=False)


def _install_vllm_transformers_processor_stub() -> None:
    if not _TEXT_ONLY_RUNTIME:
        return
    def _unsupported(*args, **kwargs):
        raise RuntimeError("multimodal processor unavailable in VLLM_TEXT_ONLY_RUNTIME")
    _install_stub_module(
        "vllm.transformers_utils.processor",
        {
            "get_processor": _unsupported,
            "cached_get_processor": _unsupported,
            "cached_processor_from_config": _unsupported,
        },
    )


def _install_vllm_kernel_warmup_stub() -> None:
    if not _TEXT_ONLY_RUNTIME:
        return
    _install_stub_module(
        "vllm.model_executor.warmup.kernel_warmup",
        {
            "kernel_warmup": (lambda *args, **kwargs: None),
            "__all__": ["kernel_warmup"],
        },
    )


def _install_vllm_gpu_metrics_logits_stub() -> None:
    if not _TEXT_ONLY_RUNTIME:
        return

    def _get_num_nans(logits):
        try:
            import torch

            return torch.zeros(logits.shape[0], dtype=torch.int32, device=logits.device)
        except Exception:
            return None

    _install_stub_module(
        "vllm.v1.worker.gpu.metrics.logits",
        {
            "get_num_nans": _get_num_nans,
            "__all__": ["get_num_nans"],
        },
    )


def _install_vllm_compilation_decorators_stub() -> None:
    if not _TEXT_ONLY_RUNTIME:
        return

    def _identity_class_decorator(cls=None, *args, **kwargs):
        if cls is None:
            def _wrap(inner):
                return inner
            return _wrap
        return cls

    _install_stub_module(
        "vllm.compilation.decorators",
        {
            "support_torch_compile": _identity_class_decorator,
            "ignore_torch_compile": _identity_class_decorator,
            "should_torch_compile_mm_encoder": (lambda *args, **kwargs: False),
            "IGNORE_COMPILE_KEY": "_ignore_compile_vllm",
            "__all__": [
                "support_torch_compile",
                "ignore_torch_compile",
                "should_torch_compile_mm_encoder",
                "IGNORE_COMPILE_KEY",
            ],
        },
    )


def _install_vllm_aiter_ops_stub() -> None:
    if not _TEXT_ONLY_RUNTIME:
        return

    class _RocmAiterOps:
        @staticmethod
        def is_enabled() -> bool:
            return False

        @staticmethod
        def refresh_env_variables(*args, **kwargs) -> None:
            return None

        @staticmethod
        def register_ops_once(*args, **kwargs) -> None:
            return None

        @staticmethod
        def initialize_aiter_allreduce(*args, **kwargs) -> None:
            return None

        @staticmethod
        def destroy_aiter_allreduce(*args, **kwargs) -> None:
            return None

        def __getattr__(self, name: str):
            if name.startswith("is_") or name.startswith("are_"):
                return lambda *args, **kwargs: False
            if name.startswith("get_"):
                return lambda *args, **kwargs: None
            if name.startswith("initialize_") or name.startswith("destroy_") or name.startswith("register_"):
                return lambda *args, **kwargs: None
            return lambda *args, **kwargs: None

    _install_stub_module(
        "vllm._aiter_ops",
        {
            "rocm_aiter_ops": _RocmAiterOps(),
            "check_aiter_fused_qk_rmsnorm": (lambda *args, **kwargs: False),
            "__all__": ["rocm_aiter_ops", "check_aiter_fused_qk_rmsnorm"],
        },
    )


def _install_vllm_distributed_comm_stub() -> None:
    if not _TEXT_ONLY_RUNTIME:
        return
    if "vllm.distributed.communication_op" in sys.modules:
        return

    module = types.ModuleType("vllm.distributed.communication_op")
    module.__dict__["__all__"] = [
        "tensor_model_parallel_all_reduce",
        "tensor_model_parallel_all_gather",
        "tensor_model_parallel_reduce_scatter",
        "tensor_model_parallel_gather",
        "broadcast_tensor_dict",
    ]

    def tensor_model_parallel_all_reduce(input_, *args, **kwargs):
        return input_

    def tensor_model_parallel_all_gather(input_, dim=-1, *args, **kwargs):
        return input_

    def tensor_model_parallel_reduce_scatter(input_, dim=-1, *args, **kwargs):
        return input_

    def tensor_model_parallel_gather(input_, dst=0, dim=-1, *args, **kwargs):
        return input_

    def broadcast_tensor_dict(tensor_dict=None, src=0, *args, **kwargs):
        return tensor_dict

    module.tensor_model_parallel_all_reduce = tensor_model_parallel_all_reduce
    module.tensor_model_parallel_all_gather = tensor_model_parallel_all_gather
    module.tensor_model_parallel_reduce_scatter = tensor_model_parallel_reduce_scatter
    module.tensor_model_parallel_gather = tensor_model_parallel_gather
    module.broadcast_tensor_dict = broadcast_tensor_dict
    sys.modules[module.__name__] = module


def _vllm_platform_package_paths() -> list[str]:
    paths: list[str] = []
    for site_dir in site.getsitepackages():
        candidate = Path(site_dir) / "vllm" / "platforms"
        if candidate.is_dir():
            paths.append(str(candidate))
    return paths


def _install_vllm_platforms_stub() -> None:
    if not _TEXT_ONLY_RUNTIME:
        return

    class PlatformEnum:
        CUDA = "cuda"
        ROCM = "rocm"
        TPU = "tpu"
        XPU = "xpu"
        CPU = "cpu"
        OOT = "oot"
        UNSPECIFIED = "unspecified"

    class CpuArchEnum:
        X86 = "x86"
        ARM = "arm"
        POWERPC = "powerpc"
        S390X = "s390x"
        RISCV = "riscv"
        OTHER = "other"
        UNKNOWN = "unknown"

    class DeviceCapability(tuple):
        __slots__ = ()

        def __new__(cls, major: int, minor: int):
            return tuple.__new__(cls, (major, minor))

        @property
        def major(self) -> int:
            return self[0]

        @property
        def minor(self) -> int:
            return self[1]

        def as_version_str(self) -> str:
            return f"{self.major}.{self.minor}"

        def to_int(self) -> int:
            return self.major * 10 + self.minor

    class Platform:
        _enum = PlatformEnum.UNSPECIFIED
        device_name = "unspecified"
        device_type = ""
        dispatch_key = "CPU"
        ray_device_key = ""
        device_control_env_var = "CUDA_VISIBLE_DEVICES"
        ray_noset_device_env_vars: list[str] = []
        simple_compile_backend = "eager"
        dist_backend = ""
        supported_quantization: list[str] = []
        additional_env_vars: list[str] = []

        @property
        def pass_key(self) -> str:
            return "post_grad_custom_post_pass"

        @property
        def supported_dtypes(self):
            import torch
            return [torch.bfloat16, torch.float16, torch.float32]

        def is_cuda(self) -> bool:
            return self._enum == PlatformEnum.CUDA

        def is_rocm(self) -> bool:
            return self._enum == PlatformEnum.ROCM

        def is_tpu(self) -> bool:
            return self._enum == PlatformEnum.TPU

        def is_xpu(self) -> bool:
            return self._enum == PlatformEnum.XPU

        def is_cpu(self) -> bool:
            return self._enum == PlatformEnum.CPU

        def is_zen_cpu(self) -> bool:
            return False

        def is_out_of_tree(self) -> bool:
            return self._enum == PlatformEnum.OOT

        def is_unspecified(self) -> bool:
            return self._enum == PlatformEnum.UNSPECIFIED

        def is_cuda_alike(self) -> bool:
            return self._enum in (PlatformEnum.CUDA, PlatformEnum.ROCM)

        def uses_host_device_handling(self) -> bool:
            return False

        def is_sleep_mode_available(self) -> bool:
            return self.is_cuda_alike()

        @classmethod
        def get_compile_backend(cls) -> str:
            return cls.simple_compile_backend

        @classmethod
        def get_pass_manager_cls(cls) -> str:
            return "vllm.compilation.passes.pass_manager.PostGradPassManager"

        @classmethod
        def import_ir_kernels(cls) -> None:
            return None

        @classmethod
        def import_kernels(cls) -> None:
            return None

        @classmethod
        def pre_register_and_update(cls, parser=None) -> None:
            return None

        @classmethod
        def apply_config_platform_defaults(cls, vllm_config) -> None:
            return None

        @classmethod
        def check_and_update_config(cls, vllm_config) -> None:
            return None

        @classmethod
        def get_device_capability(cls, device_id: int = 0):
            return None

        @classmethod
        def has_device_capability(cls, capability, device_id: int = 0) -> bool:
            current = cls.get_device_capability(device_id=device_id)
            if current is None:
                return False
            if isinstance(capability, tuple):
                return tuple(current) >= tuple(capability)
            return current.to_int() >= capability

        @classmethod
        def is_device_capability(cls, capability, device_id: int = 0) -> bool:
            current = cls.get_device_capability(device_id=device_id)
            if current is None:
                return False
            if isinstance(capability, tuple):
                return tuple(current) == tuple(capability)
            return current.to_int() == capability

        @classmethod
        def is_device_capability_family(cls, capability: int, device_id: int = 0) -> bool:
            current = cls.get_device_capability(device_id=device_id)
            if current is None:
                return False
            return (current.to_int() // 10) == (capability // 10)

        @classmethod
        def get_device_name(cls, device_id: int = 0) -> str:
            return cls.device_name

        @classmethod
        def get_device_uuid(cls, device_id: int = 0) -> str:
            return f"{cls.device_type}:{device_id}"

        @classmethod
        def get_device_total_memory(cls, device_id: int = 0) -> int:
            raise NotImplementedError

        @classmethod
        def device_count(cls) -> int:
            return 0

        @classmethod
        def inference_mode(cls):
            import torch
            return torch.inference_mode(mode=True)

        @classmethod
        def set_device(cls, device) -> None:
            raise NotImplementedError

        @classmethod
        def manual_seed_all(cls, seed: int) -> None:
            raise NotImplementedError

        @classmethod
        def get_device_communicator_cls(cls) -> str:
            return "vllm.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase"

        @classmethod
        def verify_quantization(cls, quant: str) -> None:
            if cls.supported_quantization and quant not in cls.supported_quantization:
                raise ValueError(f"{quant} quantization is currently not supported in {cls.device_name}.")

        @classmethod
        def get_cpu_architecture(cls) -> str:
            machine = os.uname().machine.lower() if hasattr(os, "uname") else ""
            if machine in ("x86_64", "amd64", "i386", "i686"):
                return CpuArchEnum.X86
            if machine.startswith("arm") or machine.startswith("aarch"):
                return CpuArchEnum.ARM
            if machine.startswith("ppc"):
                return CpuArchEnum.POWERPC
            if machine == "s390x":
                return CpuArchEnum.S390X
            if machine.startswith("riscv"):
                return CpuArchEnum.RISCV
            return CpuArchEnum.OTHER if machine else CpuArchEnum.UNKNOWN

        @classmethod
        def is_pin_memory_available(cls) -> bool:
            return True

        @classmethod
        def use_custom_allreduce(cls) -> bool:
            return False

        @classmethod
        def use_custom_op_collectives(cls) -> bool:
            return False

        @classmethod
        def is_integrated_gpu(cls, device_id: int = 0) -> bool:
            return False

        @classmethod
        def support_hybrid_kv_cache(cls) -> bool:
            return True

        @classmethod
        def support_static_graph_mode(cls) -> bool:
            return False

        @classmethod
        def supports_mx(cls) -> bool:
            return False

        @classmethod
        def supports_fp8(cls) -> bool:
            return True

        @classmethod
        def is_fp8_fnuz(cls) -> bool:
            return False

        @classmethod
        def opaque_attention_op(cls) -> bool:
            return False

        @classmethod
        def get_default_ir_op_priority(cls, vllm_config):
            from vllm.config.kernel import IrOpPriorityConfig
            return IrOpPriorityConfig.with_default(["vllm_c", "native"])

        @classmethod
        def get_attn_backend_cls(cls, selected_backend, attn_selector_config, num_heads=None) -> str:
            from vllm.v1.attention.backends.registry import AttentionBackendEnum

            if selected_backend is not None and hasattr(selected_backend, "get_path"):
                return selected_backend.get_path()
            if getattr(attn_selector_config, "use_mla", False):
                return AttentionBackendEnum.FLASHINFER_MLA.get_path()
            return AttentionBackendEnum.FLASHINFER.get_path()

        @classmethod
        def get_supported_vit_attn_backends(cls):
            from vllm.v1.attention.backends.registry import AttentionBackendEnum
            return [
                AttentionBackendEnum.FLASHINFER,
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.TORCH_SDPA,
            ]

        @classmethod
        def get_vit_attn_backend(cls, head_size, dtype, backend=None):
            from vllm.v1.attention.backends.registry import AttentionBackendEnum
            if backend is not None:
                return backend
            return AttentionBackendEnum.FLASHINFER

        @classmethod
        def set_additional_forward_context(cls, **kwargs):
            return {}

        def __getattr__(self, key: str):
            if key.startswith("__") and key.endswith("__"):
                raise AttributeError(key)
            if key.startswith(("is_", "supports_", "use_", "has_")):
                return lambda *args, **kwargs: False
            if key.startswith("update_"):
                return lambda *args, **kwargs: None
            if key.startswith("verify_"):
                return lambda *args, **kwargs: None
            if key.startswith("validate_"):
                return lambda *args, **kwargs: None
            if key.startswith("get_"):
                return lambda *args, **kwargs: None
            import torch
            device = getattr(torch, self.device_type, None)
            if device is not None and hasattr(device, key):
                attr = getattr(device, key)
                if attr is not None:
                    return attr
            raise AttributeError(key)

    class _VoicePipelineLightCudaPlatform(Platform):
        _enum = PlatformEnum.CUDA
        device_name = "NVIDIA RTX 2000 Ada Generation"
        device_type = "cuda"
        dispatch_key = "CUDA"
        ray_device_key = "GPU"
        dist_backend = "nccl"
        device_control_env_var = "CUDA_VISIBLE_DEVICES"
        ray_noset_device_env_vars = ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"]
        simple_compile_backend = "eager"

        @classmethod
        def get_device_capability(cls, device_id: int = 0):
            return DeviceCapability(8, 9)

        @classmethod
        def get_device_total_memory(cls, device_id: int = 0) -> int:
            return 16380 * 1024 * 1024

        @classmethod
        def device_count(cls) -> int:
            return 2

        @classmethod
        def set_device(cls, device) -> None:
            import torch
            torch.cuda.set_device(device)

        @classmethod
        def manual_seed_all(cls, seed: int) -> None:
            import torch
            torch.cuda.manual_seed_all(seed)

        @classmethod
        def fp8_dtype(cls):
            import torch
            return torch.float8_e4m3fn

        @classmethod
        def check_if_supports_dtype(cls, dtype):
            import torch
            if dtype == torch.bfloat16 and not cls.has_device_capability(80):
                raise ValueError("bfloat16 requires compute capability >= 8.0")

        @classmethod
        def check_and_update_config(cls, vllm_config) -> None:
            parallel_config = vllm_config.parallel_config
            if getattr(parallel_config, "worker_cls", None) == "auto":
                parallel_config.worker_cls = "vllm.v1.worker.gpu_worker.Worker"
            model_config = vllm_config.model_config
            scheduler_config = vllm_config.scheduler_config
            if (
                model_config is not None
                and getattr(model_config, "is_mm_prefix_lm", False)
                and getattr(scheduler_config, "is_multimodal_model", False)
                and not getattr(scheduler_config, "disable_chunked_mm_input", False)
            ):
                scheduler_config.disable_chunked_mm_input = True

    current_platform = _VoicePipelineLightCudaPlatform()
    _install_stub_module(
        "vllm.platforms",
        {
            "__path__": _vllm_platform_package_paths(),
            "PlatformEnum": PlatformEnum,
            "CpuArchEnum": CpuArchEnum,
            "DeviceCapability": DeviceCapability,
            "Platform": Platform,
            "current_platform": current_platform,
            "_current_platform": current_platform,
            "_init_trace": "voice_pipeline: stubbed lightweight cuda platform",
        },
        is_package=True,
    )


def _install_vllm_config_device_stub() -> None:
    if not _TEXT_ONLY_RUNTIME:
        return

    Device = str

    @dataclass
    class DeviceConfig:
        device: Any = "auto"
        device_type: str = field(init=False)

        def compute_hash(self) -> str:
            return "voice-pipeline-text-only-device"

        def __post_init__(self) -> None:
            if self.device == "auto":
                self.device_type = "cuda"
            elif isinstance(self.device, str):
                self.device_type = self.device
            else:
                self.device_type = getattr(self.device, "type", str(self.device))
            self.device = self.device_type

    _install_stub_module(
        "vllm.config.device",
        {
            "Device": Device,
            "DeviceConfig": DeviceConfig,
            "_TEXT_ONLY_RUNTIME": True,
        },
    )


def _install_vllm_mcp_tool_server_stub() -> None:
    if not _TEXT_ONLY_RUNTIME:
        return
    module_name = "vllm.entrypoints.mcp.tool_server"
    if module_name in sys.modules:
        return

    class _ToolServer:
        def __init__(self, *args, **kwargs):
            pass

    module = types.ModuleType(module_name)
    module.ToolServer = _ToolServer
    sys.modules[module_name] = module


def _install_pydantic_plugin_loader_stub() -> None:
    if not _TEXT_ONLY_RUNTIME:
        return
    module_name = "pydantic.plugin._loader"
    if module_name in sys.modules:
        return
    module = types.ModuleType(module_name)
    module.PYDANTIC_ENTRY_POINT_GROUP = "pydantic"
    module._plugins = {}
    module._loading_plugins = False
    module.get_plugins = lambda: ()
    sys.modules[module_name] = module


def _install_transformers_text_only_stubs() -> None:
    if not _TEXT_ONLY_RUNTIME:
        return
    _install_transformers_root_stub()

    class _UnavailableAutoComponent:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise RuntimeError(f"{cls.__name__} unavailable in TRANSFORMERS_TEXT_ONLY_RUNTIME")

    stub_specs: dict[str, dict[str, Any]] = {
        "transformers.models.auto.feature_extraction_auto": {
            "AutoFeatureExtractor": type("AutoFeatureExtractor", (_UnavailableAutoComponent,), {}),
            "FEATURE_EXTRACTOR_MAPPING": {},
            "FEATURE_EXTRACTOR_MAPPING_NAMES": {},
            "get_feature_extractor_config": lambda *args, **kwargs: {},
        },
        "transformers.models.auto.image_processing_auto": {
            "AutoImageProcessor": type("AutoImageProcessor", (_UnavailableAutoComponent,), {}),
            "IMAGE_PROCESSOR_MAPPING": {},
            "IMAGE_PROCESSOR_MAPPING_NAMES": {},
            "get_image_processor_config": lambda *args, **kwargs: {},
        },
        "transformers.models.auto.processing_auto": {
            "AutoProcessor": type("AutoProcessor", (_UnavailableAutoComponent,), {}),
            "PROCESSOR_MAPPING": {},
            "PROCESSOR_MAPPING_NAMES": {},
            "processor_class_from_name": lambda *args, **kwargs: None,
        },
        "transformers.models.auto.video_processing_auto": {
            "AutoVideoProcessor": type("AutoVideoProcessor", (_UnavailableAutoComponent,), {}),
            "VIDEO_PROCESSOR_MAPPING": {},
            "VIDEO_PROCESSOR_MAPPING_NAMES": {},
        },
    }
    for module_name, attrs in stub_specs.items():
        _install_stub_module(module_name, attrs)
    _install_transformers_masking_utils_stub()


def _install_transformers_masking_utils_stub() -> None:
    if not _TEXT_ONLY_RUNTIME:
        return

    def _return_attention_mask(*args, **kwargs):
        return kwargs.get("attention_mask")

    def _return_generate_masks(*args, **kwargs):
        return kwargs.get("attention_mask")

    _install_stub_module(
        "transformers.masking_utils",
        {
            "BlockMask": object,
            "create_causal_mask": _return_attention_mask,
            "create_sliding_window_causal_mask": _return_attention_mask,
            "create_chunked_causal_mask": _return_attention_mask,
            "create_masks_for_generate": _return_generate_masks,
        },
    )


def _transformers_package_paths() -> list[str]:
    paths: list[str] = []
    for site_dir in site.getsitepackages():
        candidate = Path(site_dir) / "transformers"
        if candidate.is_dir():
            paths.append(str(candidate))
    return paths


def _install_transformers_root_stub() -> None:
    if not _TEXT_ONLY_RUNTIME:
        return

    paths = _transformers_package_paths()
    version_str = "0"
    try:
        version_str = version("transformers")
    except Exception:
        pass

    attr_map = {
        "PretrainedConfig": ("transformers.configuration_utils", "PretrainedConfig"),
        "GenerationConfig": ("transformers.generation.configuration_utils", "GenerationConfig"),
        "AutoConfig": ("transformers.models.auto.configuration_auto", "AutoConfig"),
        "AutoTokenizer": ("transformers.models.auto.tokenization_auto", "AutoTokenizer"),
        "AutoModelForCausalLM": ("transformers.models.auto.modeling_auto", "AutoModelForCausalLM"),
        "AutoModelForImageTextToText": (
            "transformers.models.auto.modeling_auto",
            "AutoModelForImageTextToText",
        ),
        "AutoFeatureExtractor": ("transformers.models.auto.feature_extraction_auto", "AutoFeatureExtractor"),
        "AutoImageProcessor": ("transformers.models.auto.image_processing_auto", "AutoImageProcessor"),
        "AutoProcessor": ("transformers.models.auto.processing_auto", "AutoProcessor"),
        "AutoVideoProcessor": ("transformers.models.auto.video_processing_auto", "AutoVideoProcessor"),
        "AddedToken": ("transformers.tokenization_utils_base", "AddedToken"),
        "BatchEncoding": ("transformers.tokenization_utils_base", "BatchEncoding"),
        "LogitsProcessor": ("transformers.generation", "LogitsProcessor"),
        "LogitsProcessorList": ("transformers.generation", "LogitsProcessorList"),
        "PreTrainedTokenizerBase": ("transformers.tokenization_utils_base", "PreTrainedTokenizerBase"),
        "PreTrainedTokenizerFast": ("transformers.tokenization_utils_fast", "PreTrainedTokenizerFast"),
        "PreTrainedTokenizer": ("transformers.tokenization_utils", "PreTrainedTokenizer"),
        "BatchFeature": ("transformers.feature_extraction_utils", "BatchFeature"),
        "ProcessorMixin": ("transformers.processing_utils", "ProcessorMixin"),
        "TensorType": ("transformers.utils.generic", "TensorType"),
        "BaseImageProcessor": ("transformers.image_processing_utils", "BaseImageProcessor"),
        "CONFIG_MAPPING": ("transformers.models.auto.configuration_auto", "CONFIG_MAPPING"),
        "Qwen2Config": ("transformers.models.qwen2.configuration_qwen2", "Qwen2Config"),
        "Qwen3Config": ("transformers.models.qwen3.configuration_qwen3", "Qwen3Config"),
        "Qwen2Model": ("transformers.models.qwen2.modeling_qwen2", "Qwen2Model"),
        "Qwen2ForCausalLM": ("transformers.models.qwen2.modeling_qwen2", "Qwen2ForCausalLM"),
        "WhisperConfig": ("transformers.models.whisper.configuration_whisper", "WhisperConfig"),
        "LlamaConfig": ("transformers.models.llama.configuration_llama", "LlamaConfig"),
        "SiglipVisionConfig": ("transformers.models.siglip.configuration_siglip", "SiglipVisionConfig"),
        "DeepseekV2Config": ("transformers.models.deepseek_v2.configuration_deepseek_v2", "DeepseekV2Config"),
        "DeepseekV3Config": ("transformers.models.deepseek_v3.configuration_deepseek_v3", "DeepseekV3Config"),
        "ModernBertConfig": ("transformers.models.modernbert.configuration_modernbert", "ModernBertConfig"),
        "PaliGemmaConfig": ("transformers.models.paligemma.configuration_paligemma", "PaliGemmaConfig"),
        "ParakeetEncoderConfig": ("transformers.models.parakeet.configuration_parakeet", "ParakeetEncoderConfig"),
        "Qwen2VLConfig": ("transformers.models.qwen2_vl.configuration_qwen2_vl", "Qwen2VLConfig"),
    }

    def __getattr__(name: str):
        target = attr_map.get(name)
        if target is None:
            raise AttributeError(name)
        module_name, attr_name = target
        import importlib

        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
        sys.modules["transformers"].__dict__[name] = value
        return value

    _install_stub_module(
        "transformers",
        {
            "__path__": paths,
            "__version__": version_str,
            "__getattr__": __getattr__,
        },
        is_package=True,
    )


def _patch_torch_cuda(module: Any) -> None:
    if getattr(module, "__voice_pipeline_lazy_call_patched__", False):
        return
    if os.getenv("TORCH_LIBRARY_SKIP_TRACEBACK", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return

    original_lazy_call = module._lazy_call

    def _patched_lazy_call(callable, **kwargs):
        import traceback

        with module._initialization_lock:
            if module.is_initialized():
                callable()
                return
            stack = []
            if kwargs.get("seed_all", False):
                module._lazy_seed_tracker.queue_seed_all(callable, stack)
            elif kwargs.get("seed", False):
                module._lazy_seed_tracker.queue_seed(callable, stack)
            else:
                module._queued_calls.append((callable, stack))

    module._lazy_call = _patched_lazy_call
    module.__voice_pipeline_lazy_call_patched__ = True


def _patch_torch_root(module: Any) -> None:
    if not _TEXT_ONLY_RUNTIME or getattr(module, "__voice_pipeline_compile_patched__", False):
        return

    def _identity_compile(fn=None, *args, **kwargs):
        if fn is None:
            def _wrap(inner):
                return inner
            return _wrap
        return fn

    module.compile = _identity_compile
    module.__voice_pipeline_compile_patched__ = True


def _patch_vllm_config_device(module: Any) -> None:
    if not _TEXT_ONLY_RUNTIME:
        return
    device_config_cls = getattr(module, "DeviceConfig", None)
    if device_config_cls is None or getattr(device_config_cls, "__voice_pipeline_patched__", False):
        return

    def __post_init__(self):
        if self.device == "auto":
            self.device_type = "cuda"
        else:
            if isinstance(self.device, str):
                self.device_type = self.device
            else:
                self.device_type = getattr(self.device, "type", str(self.device))
        self.device = self.device_type

    device_config_cls.__post_init__ = __post_init__
    device_config_cls.__voice_pipeline_patched__ = True


def _patch_vllm_request(module: Any) -> None:
    if not _TEXT_ONLY_RUNTIME:
        return
    module.MultiModalFeatureSpec = Any
    module.StructuredOutputRequest = None


def _patch_vllm_plugins(module: Any) -> None:
    if not _TEXT_ONLY_RUNTIME:
        return

    def _no_plugins(*args, **kwargs):
        return {}

    module.load_general_plugins = lambda: None
    module.load_plugins_by_group = _no_plugins


def _patch_vllm_custom_ops(module: Any) -> None:
    if not _TEXT_ONLY_RUNTIME:
        return

    def _call_support_op(name: str, cuda_device_capability: int) -> bool:
        try:
            import torch
            return bool(getattr(torch.ops._C, name)(cuda_device_capability))
        except Exception:
            return False

    module.cutlass_scaled_mm_supports_fp8 = (
        lambda cuda_device_capability: _call_support_op(
            "cutlass_scaled_mm_supports_fp8", cuda_device_capability
        )
    )
    module.cutlass_scaled_mm_supports_block_fp8 = (
        lambda cuda_device_capability: _call_support_op(
            "cutlass_scaled_mm_supports_block_fp8", cuda_device_capability
        )
    )


def _patch_vllm_activation_layer(module: Any) -> None:
    if not _TEXT_ONLY_RUNTIME:
        return

    import torch

    def _missing_op(name: str) -> bool:
        try:
            getattr(torch.ops._C, name)
            return False
        except Exception:
            return True

    def _native_only_init_without_args(cls: type[Any], op_name: str) -> None:
        original_init = cls.__init__
        if getattr(original_init, "__voice_pipeline_patched__", False):
            return

        def _patched(self, *args, **kwargs):
            if _missing_op(op_name):
                super(cls, self).__init__(*args, **kwargs)
                self._forward_method = self.forward_native
                return
            original_init(self, *args, **kwargs)

        _patched.__voice_pipeline_patched__ = True
        cls.__init__ = _patched

    def _native_only_init_with_threshold(cls: type[Any], op_name: str) -> None:
        original_init = cls.__init__
        if getattr(original_init, "__voice_pipeline_patched__", False):
            return

        def _patched(self, threshold=0.0):
            if _missing_op(op_name):
                super(cls, self).__init__()
                self.threshold = threshold
                self._forward_method = self.forward_native
                return
            original_init(self, threshold)

        _patched.__voice_pipeline_patched__ = True
        cls.__init__ = _patched

    def _native_only_init_with_limit(cls: type[Any], op_name: str) -> None:
        original_init = cls.__init__
        if getattr(original_init, "__voice_pipeline_patched__", False):
            return

        def _patched(self, swiglu_limit: float, *, compile_native: bool = True):
            if _missing_op(op_name):
                super(cls, self).__init__(compile_native=compile_native)
                self.swiglu_limit = float(swiglu_limit)
                self._forward_method = self.forward_native
                return
            original_init(self, swiglu_limit, compile_native=compile_native)

        _patched.__voice_pipeline_patched__ = True
        cls.__init__ = _patched

    _native_only_init_with_threshold(module.FatreluAndMul, "fatrelu_and_mul")
    _native_only_init_without_args(module.SiluAndMul, "silu_and_mul")
    _native_only_init_with_limit(
        module.SiluAndMulWithClamp, "silu_and_mul_with_clamp"
    )
    _native_only_init_without_args(module.MulAndSilu, "mul_and_silu")


def _patch_vllm_kernel_vllm_c(module: Any) -> None:
    if not _TEXT_ONLY_RUNTIME:
        return

    import torch

    def _has_op(name: str) -> bool:
        try:
            getattr(torch.ops._C, name)
            return True
        except Exception:
            return False

    if not _has_op("rms_norm"):
        native_rms = module.ir.ops.rms_norm.impls["native"].impl_fn

        def _patched_rms_norm(x, weight, epsilon, variance_size=None):
            return native_rms(x, weight, epsilon, variance_size)

        module.rms_norm = _patched_rms_norm
        if "vllm_c" in module.ir.ops.rms_norm.impls:
            module.ir.ops.rms_norm.impls["vllm_c"].impl_fn = _patched_rms_norm

    if not _has_op("fused_add_rms_norm"):
        native_fused = module.ir.ops.fused_add_rms_norm.impls["native"].impl_fn

        def _patched_fused_add_rms_norm(
            x,
            x_residual,
            weight,
            epsilon,
            variance_size=None,
        ):
            return native_fused(x, x_residual, weight, epsilon, variance_size)

        module.fused_add_rms_norm = _patched_fused_add_rms_norm
        if "vllm_c" in module.ir.ops.fused_add_rms_norm.impls:
            module.ir.ops.fused_add_rms_norm.impls["vllm_c"].impl_fn = (
                _patched_fused_add_rms_norm
            )


def _patch_vllm_rotary_embedding_base(module: Any) -> None:
    if not _TEXT_ONLY_RUNTIME:
        return

    import torch

    def _has_rotary_op() -> bool:
        try:
            getattr(torch.ops._C, "rotary_embedding")
            return True
        except Exception:
            return False

    cls = getattr(module, "RotaryEmbedding", None)
    if cls is None:
        return

    original_cuda = cls.forward_cuda
    if not getattr(original_cuda, "__voice_pipeline_patched__", False):
        def _patched_forward_cuda(self, positions, query, key=None):
            if not _has_rotary_op():
                return self.forward_native(positions, query, key)
            return original_cuda(self, positions, query, key)

        _patched_forward_cuda.__voice_pipeline_patched__ = True
        cls.forward_cuda = _patched_forward_cuda

    original_cpu = cls.forward_cpu
    if not getattr(original_cpu, "__voice_pipeline_patched__", False):
        def _patched_forward_cpu(self, positions, query, key=None):
            if not _has_rotary_op():
                return self.forward_native(positions, query, key)
            return original_cpu(self, positions, query, key)

        _patched_forward_cpu.__voice_pipeline_patched__ = True
        cls.forward_cpu = _patched_forward_cpu


def _patch_vllm_input_quant_fp8(module: Any) -> None:
    if not _TEXT_ONLY_RUNTIME:
        return

    import torch

    def _has_needed_fp8_ops(static: bool) -> bool:
        needed = (
            ["static_scaled_fp8_quant"]
            if static
            else ["dynamic_per_token_scaled_fp8_quant", "dynamic_scaled_fp8_quant"]
        )
        try:
            for name in needed:
                getattr(torch.ops._C, name)
            return True
        except Exception:
            return False

    cls = getattr(module, "QuantFP8", None)
    if cls is None:
        return
    original_cuda = cls.forward_cuda
    if getattr(original_cuda, "__voice_pipeline_patched__", False):
        return

    def _patched_forward_cuda(
        self,
        x,
        scale=None,
        scale_ub=None,
        use_triton: bool = False,
    ):
        if not _has_needed_fp8_ops(self.static):
            return self.forward_native(x, scale, scale_ub, use_triton)
        return original_cuda(self, x, scale, scale_ub, use_triton)

    _patched_forward_cuda.__voice_pipeline_patched__ = True
    cls.forward_cuda = _patched_forward_cuda


def _patch_vllm_gpu_buffer_utils(module: Any) -> None:
    if not _TEXT_ONLY_RUNTIME:
        return

    def _has_cuda_uva_view_op() -> bool:
        try:
            import torch
            getattr(torch.ops._C, "get_cuda_view_from_cpu_tensor")
            return True
        except Exception:
            return False

    if _has_cuda_uva_view_op():
        return

    import torch

    def _fallback_device() -> torch.device:
        try:
            return torch.device("cuda", torch.cuda.current_device())
        except Exception:
            return torch.device("cuda:0")

    original_uva_init = module.UvaBuffer.__init__

    def _patched_uva_init(self, size, dtype):
        try:
            original_uva_init(self, size, dtype)
            self._voice_pipeline_fallback_copy = False
        except Exception:
            self.cpu = torch.zeros(size, dtype=dtype, device="cpu", pin_memory=False)
            self.np = self.cpu.numpy()
            self.uva = module.async_copy_to_gpu(self.cpu, device=_fallback_device())
            self._voice_pipeline_fallback_copy = True

    def _patched_pool_copy_to_uva(self, x):
        self._curr = (self._curr + 1) % self.max_concurrency
        buf = self._uva_bufs[self._curr]
        dst = buf.cpu if isinstance(x, torch.Tensor) else buf.np
        n = len(x)
        dst[:n] = x
        if getattr(buf, "_voice_pipeline_fallback_copy", False):
            return module.async_copy_to_gpu(buf.cpu[:n], device=_fallback_device())
        return buf.uva[:n]

    def _patched_tensor_copy_to_uva(self, n=None):
        view = self.np[:n] if n is not None else self.np
        self.gpu = self.pool.copy_to_uva(view)
        return self.gpu

    module.UvaBuffer.__init__ = _patched_uva_init
    module.UvaBufferPool.copy_to_uva = _patched_pool_copy_to_uva
    module.UvaBackedTensor.copy_to_uva = _patched_tensor_copy_to_uva


def _patch_vllm_mem_utils(module: Any) -> None:
    if not _TEXT_ONLY_RUNTIME:
        return

    profiler_cls = getattr(module, "DeviceMemoryProfiler", None)
    if profiler_cls is None:
        return
    original_exit = profiler_cls.__exit__
    if getattr(original_exit, "__voice_pipeline_patched__", False):
        return

    def _patched_exit(self, exc_type, exc_val, exc_tb):
        try:
            self.final_memory = self.current_memory_usage()
        except Exception:
            self.final_memory = None
        initial = getattr(self, "initial_memory", None)
        final = getattr(self, "final_memory", None)
        if initial is None or final is None:
            self.consumed_memory = 0
        else:
            self.consumed_memory = final - initial
        try:
            import gc

            gc.collect()
        except Exception:
            pass
        return None

    _patched_exit.__voice_pipeline_patched__ = True
    profiler_cls.__exit__ = _patched_exit


def _patch_vllm_outputs(module: Any) -> None:
    if not _TEXT_ONLY_RUNTIME:
        return
    module.CUDAGraphStat = Any


def _patch_vllm_async_llm(module: Any) -> None:
    if not _TEXT_ONLY_RUNTIME:
        return

    def _default_mm_registry():
        return None

    module._default_mm_registry = _default_mm_registry


def _patch_vllm_import_utils(module: Any) -> None:
    if not _TEXT_ONLY_RUNTIME:
        return
    module.has_deep_gemm = lambda: False


def _patch_vllm_transformers_config(module: Any) -> None:
    if not _TEXT_ONLY_RUNTIME:
        return

    class _LightPretrainedConfig:
        def __init__(self, **kwargs) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

        @classmethod
        def get_config_dict(
            cls,
            model: str | Path,
            revision: str | None = None,
            code_revision: str | None = None,
            **kwargs,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            config_path = Path(model) / "config.json"
            if not config_path.is_file():
                raise FileNotFoundError(f"Missing config.json for model path: {model}")
            with config_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return payload, kwargs

    def _skip_speculators(
        model: str | Path,
        tokenizer: str | None,
        trust_remote_code: bool,
        revision: str | None = None,
        vllm_speculative_config: dict[str, Any] | None = None,
        hf_token: str | None = None,
        **kwargs,
    ) -> tuple[str | Path, str | None, dict[str, Any] | None]:
        return model, tokenizer, vllm_speculative_config

    def _hf_config_name() -> str:
        return "config.json"

    def _pretrained_config_cls():
        return _LightPretrainedConfig

    def _allowed_attention_layer_types():
        return set()

    module.maybe_override_with_speculators = _skip_speculators
    module._hf_config_name = _hf_config_name
    module._pretrained_config_cls = _pretrained_config_cls
    module._allowed_attention_layer_types = _allowed_attention_layer_types


def _patch_vllm_platforms(module: Any) -> None:
    if not _TEXT_ONLY_RUNTIME:
        return
    if multiprocessing.current_process().name != "MainProcess":
        return
    if getattr(module, "__voice_pipeline_current_platform_patched__", False):
        return

    Platform = getattr(module, "Platform", None)
    PlatformEnum = getattr(module, "PlatformEnum", None)
    DeviceCapability = getattr(module, "DeviceCapability", None)
    if Platform is None or PlatformEnum is None or DeviceCapability is None:
        return

    class _VoicePipelineLightCudaPlatform(Platform):
        _enum = PlatformEnum.CUDA
        device_name = "cuda"
        device_type = "cuda"
        dispatch_key = "CUDA"
        ray_device_key = "GPU"
        dist_backend = "nccl"
        device_control_env_var = "CUDA_VISIBLE_DEVICES"
        ray_noset_device_env_vars = [
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
        ]
        simple_compile_backend = "eager"

        @classmethod
        def get_device_capability(cls, device_id: int = 0):
            return DeviceCapability(8, 9)

        @classmethod
        def get_device_name(cls, device_id: int = 0) -> str:
            return "NVIDIA RTX 2000 Ada Generation"

        @classmethod
        def get_device_total_memory(cls, device_id: int = 0) -> int:
            return 16380 * 1024 * 1024

        @classmethod
        def device_count(cls) -> int:
            return 2

        @classmethod
        def set_device(cls, device) -> None:
            import torch

            torch.cuda.set_device(device)

        @classmethod
        def manual_seed_all(cls, seed: int) -> None:
            import torch

            torch.cuda.manual_seed_all(seed)

        @classmethod
        def check_and_update_config(cls, vllm_config) -> None:
            parallel_config = vllm_config.parallel_config
            model_config = vllm_config.model_config
            if getattr(parallel_config, "worker_cls", None) == "auto":
                parallel_config.worker_cls = "vllm.v1.worker.gpu_worker.Worker"
            scheduler_config = vllm_config.scheduler_config
            if (
                model_config is not None
                and getattr(model_config, "is_mm_prefix_lm", False)
                and getattr(scheduler_config, "is_multimodal_model", False)
                and not getattr(scheduler_config, "disable_chunked_mm_input", False)
            ):
                scheduler_config.disable_chunked_mm_input = True

        @classmethod
        def fp8_dtype(cls):
            import torch

            return torch.float8_e4m3fn

        @classmethod
        def check_if_supports_dtype(cls, dtype):
            import torch

            if dtype == torch.bfloat16 and not cls.has_device_capability(80):
                raise ValueError("bfloat16 requires compute capability >= 8.0")

        @classmethod
        def support_hybrid_kv_cache(cls) -> bool:
            return True

        @classmethod
        def support_static_graph_mode(cls) -> bool:
            return False

        @classmethod
        def get_default_ir_op_priority(cls, vllm_config):
            from vllm.config.kernel import IrOpPriorityConfig

            return IrOpPriorityConfig.with_default(["vllm_c", "native"])

    current_platform = _VoicePipelineLightCudaPlatform()
    module._current_platform = current_platform
    module.current_platform = current_platform
    module._init_trace = "voice_pipeline: lightweight parent cuda platform"
    module.__voice_pipeline_current_platform_patched__ = True


def _patch_pydantic_plugin_loader(module: Any) -> None:
    if not _TEXT_ONLY_RUNTIME:
        return

    def _no_plugins():
        return ()

    module.get_plugins = _no_plugins
    if hasattr(module, "_plugins"):
        module._plugins = {}
    if hasattr(module, "_loading_plugins"):
        module._loading_plugins = False


def _patch_transformers_root(module: Any) -> None:
    if not _TEXT_ONLY_RUNTIME:
        return
    for attr, module_name in (
        ("AutoFeatureExtractor", "transformers.models.auto.feature_extraction_auto"),
        ("AutoImageProcessor", "transformers.models.auto.image_processing_auto"),
        ("AutoProcessor", "transformers.models.auto.processing_auto"),
        ("AutoVideoProcessor", "transformers.models.auto.video_processing_auto"),
    ):
        stub_module = sys.modules.get(module_name)
        if stub_module is None:
            continue
        value = getattr(stub_module, attr, None)
        if value is not None:
            setattr(module, attr, value)


def _install_runtime_import_patches() -> None:
    _patch_importlib_metadata()
    _install_deep_gemm_stub()
    _install_torch_dynamo_stub()
    _install_pyworld_stub()
    _install_whisper_stub()
    _install_cosyvoice_dataset_processor_stub()
    _install_vllm_platforms_stub()
    _install_vllm_multimodal_stubs()
    _install_vllm_multimodal_processing_stubs()
    _install_vllm_transformers_processor_stub()
    _install_vllm_kernel_warmup_stub()
    _install_vllm_gpu_metrics_logits_stub()
    _install_vllm_compilation_decorators_stub()
    _install_vllm_aiter_ops_stub()
    _install_vllm_distributed_comm_stub()
    _install_vllm_config_device_stub()
    _install_vllm_mcp_tool_server_stub()
    _install_pydantic_plugin_loader_stub()
    _install_transformers_text_only_stubs()
    _install_post_import_patch("torch", _patch_torch_root)
    _install_post_import_patch("torch.cuda", _patch_torch_cuda)
    _install_post_import_patch("transformers", _patch_transformers_root)
    _install_post_import_patch("vllm.plugins", _patch_vllm_plugins)
    _install_post_import_patch("vllm._custom_ops", _patch_vllm_custom_ops)
    _install_post_import_patch(
        "vllm.model_executor.layers.activation", _patch_vllm_activation_layer
    )
    _install_post_import_patch(
        "vllm.model_executor.layers.rotary_embedding.base",
        _patch_vllm_rotary_embedding_base,
    )
    _install_post_import_patch(
        "vllm.model_executor.layers.quantization.input_quant_fp8",
        _patch_vllm_input_quant_fp8,
    )
    _install_post_import_patch("vllm.kernels.vllm_c", _patch_vllm_kernel_vllm_c)
    _install_post_import_patch("vllm.v1.worker.gpu.buffer_utils", _patch_vllm_gpu_buffer_utils)
    _install_post_import_patch("vllm.utils.mem_utils", _patch_vllm_mem_utils)
    _install_post_import_patch("vllm.platforms", _patch_vllm_platforms)
    _install_post_import_patch("vllm.config.device", _patch_vllm_config_device)
    _install_post_import_patch("vllm.v1.request", _patch_vllm_request)
    _install_post_import_patch("vllm.v1.outputs", _patch_vllm_outputs)
    _install_post_import_patch("vllm.v1.engine.async_llm", _patch_vllm_async_llm)
    _install_post_import_patch("vllm.utils.import_utils", _patch_vllm_import_utils)
    _install_post_import_patch("vllm.transformers_utils.config", _patch_vllm_transformers_config)
    _install_post_import_patch("pydantic.plugin._loader", _patch_pydantic_plugin_loader)


Path.rglob = _patched_rglob
_patch_inspect_filename_guards()
_prepend_cosyvoice_runtime_paths()
_prepend_cuda_runtime_library_paths()
_prepend_runtime_sys_path_entries()
_append_runtime_sys_path_entries()
_install_runtime_import_patches()
