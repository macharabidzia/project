#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "apps" / "api" / "src" / "voice_pipeline"
SYMBOL_INVENTORY_PATH = REPO_ROOT / "docs" / "voice_pipeline_symbol_inventory.json"
DEPENDENCY_SNAPSHOT_PATH = REPO_ROOT / "docs" / "voice_pipeline_dependency_snapshot.json"
OVERVIEW_PATH = REPO_ROOT / "overview.md"


@dataclass(frozen=True)
class FunctionInfo:
    name: str
    kind: str
    doc: str
    lineno: int


@dataclass(frozen=True)
class ClassInfo:
    name: str
    doc: str
    lineno: int
    methods: tuple[FunctionInfo, ...]


@dataclass(frozen=True)
class ModuleInfo:
    path: str
    module_doc: str
    classes: tuple[ClassInfo, ...]
    functions: tuple[FunctionInfo, ...]
    imports: tuple[str, ...]
    symbol_imports: tuple[tuple[str, tuple[str, ...]], ...]


def _top_level_functions(tree: ast.Module) -> tuple[FunctionInfo, ...]:
    items: list[FunctionInfo] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            items.append(
                FunctionInfo(
                    name=str(node.name),
                    kind="async_function" if isinstance(node, ast.AsyncFunctionDef) else "function",
                    doc=str(ast.get_docstring(node) or ""),
                    lineno=int(node.lineno),
                )
            )
    return tuple(items)


def _class_infos(tree: ast.Module) -> tuple[ClassInfo, ...]:
    classes: list[ClassInfo] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        methods: list[FunctionInfo] = []
        for body in node.body:
            if isinstance(body, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(
                    FunctionInfo(
                        name=str(body.name),
                        kind="async_method" if isinstance(body, ast.AsyncFunctionDef) else "method",
                        doc=str(ast.get_docstring(body) or ""),
                        lineno=int(body.lineno),
                    )
                )
        classes.append(
            ClassInfo(
                name=str(node.name),
                doc=str(ast.get_docstring(node) or ""),
                lineno=int(node.lineno),
                methods=tuple(methods),
            )
        )
    return tuple(classes)


def _collect_imports(tree: ast.Module) -> tuple[tuple[str, ...], tuple[tuple[str, tuple[str, ...]], ...]]:
    imports: list[str] = []
    symbols: dict[str, list[str]] = defaultdict(list)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(str(alias.name or "").strip())
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            module = str(node.module or "").strip()
            if not module:
                continue
            imports.append(module)
            for alias in node.names:
                symbols[module].append(str(alias.name or "").strip())
    return tuple(sorted(set(imports))), tuple(sorted((k, tuple(v)) for k, v in symbols.items()))


def _iter_modules() -> tuple[ModuleInfo, ...]:
    modules: list[ModuleInfo] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        rel = path.relative_to(SRC_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        imports, symbol_imports = _collect_imports(tree)
        modules.append(
            ModuleInfo(
                path=rel,
                module_doc=str(ast.get_docstring(tree) or ""),
                classes=_class_infos(tree),
                functions=_top_level_functions(tree),
                imports=imports,
                symbol_imports=symbol_imports,
            )
        )
    return tuple(modules)


def _write_symbol_inventory(modules: tuple[ModuleInfo, ...]) -> None:
    payload: list[dict[str, object]] = []
    for module in modules:
        payload.append(
            {
                "path": module.path,
                "module_doc": module.module_doc,
                "classes": [
                    {
                        "name": cls.name,
                        "doc": cls.doc,
                        "lineno": cls.lineno,
                        "methods": [
                            {
                                "name": method.name,
                                "kind": method.kind,
                                "doc": method.doc,
                                "lineno": method.lineno,
                            }
                            for method in cls.methods
                        ],
                    }
                    for cls in module.classes
                ],
                "functions": [
                    {
                        "name": fn.name,
                        "kind": fn.kind,
                        "doc": fn.doc,
                        "lineno": fn.lineno,
                    }
                    for fn in module.functions
                ],
            }
        )
    SYMBOL_INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SYMBOL_INVENTORY_PATH.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _dependency_snapshot(modules: tuple[ModuleInfo, ...]) -> dict[str, object]:
    module_to_imports: dict[str, list[str]] = {}
    adapter_import_violations: list[str] = []
    reducer_import_violations: list[str] = []
    transport_authority_mutation_hits: list[str] = []

    for module in modules:
        module_name = f"voice_pipeline.{module.path[:-3].replace('/', '.')}"
        imports = [name for name in module.imports if name.startswith("voice_pipeline.")]
        module_to_imports[module_name] = sorted(set(imports))

        for target, symbols in module.symbol_imports:
            if target in {
                "voice_pipeline.worker.runtime.vllm_stream",
                "voice_pipeline.worker.runtime.tts_cosy_stream",
            } and not module.path.startswith("kernel/"):
                adapter_import_violations.append(f"{module.path} imports execution adapter module {target}")
            if target == "voice_pipeline.kernel.reducer" and not module.path.startswith("kernel/"):
                if any(symbol in {"reduce_event", "ReducerTransition", "DispatchCommand"} for symbol in symbols):
                    reducer_import_violations.append(f"{module.path} imports reducer transition symbol(s) from {target}")

        if module.path.startswith("worker/transport/"):
            source_path = SRC_ROOT / module.path
            tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and str(node.attr or "") in {
                    "authority_loop",
                    "apply_external",
                    "enqueue_authority_event",
                    "submit_cancel",
                    "submit_interrupt",
                    "reduce_event",
                }:
                    transport_authority_mutation_hits.append(
                        f"{module.path}:{int(getattr(node, 'lineno', 0) or 0)} references authority loop attribute {node.attr}"
                    )

    edges = [
        {"module": module_name, "imports": imports}
        for module_name, imports in sorted(module_to_imports.items())
    ]

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "module_edges": edges,
        "violations": {
            "execution_adapter_imports_outside_kernel": sorted(set(adapter_import_violations)),
            "reducer_transition_imports_outside_kernel": sorted(set(reducer_import_violations)),
            "transport_authority_mutation_hits": sorted(set(transport_authority_mutation_hits)),
        },
    }


def _write_dependency_snapshot(modules: tuple[ModuleInfo, ...]) -> None:
    snapshot = _dependency_snapshot(modules)
    DEPENDENCY_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEPENDENCY_SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _module_title(rel_path: str) -> str:
    if "/" in rel_path:
        section = rel_path.split("/", 1)[0]
    else:
        section = "root"
    return section.capitalize()


def _module_summary(module: ModuleInfo) -> str:
    doc = module.module_doc.strip()
    if doc:
        return doc.splitlines()[0].strip()
    return "Module with runtime support code."


def _doc_summary(doc: str) -> str:
    summary = str(doc or "").strip()
    if not summary:
        return ""
    return summary.splitlines()[0].strip().rstrip(".")


def _verb_phrase(name: str) -> str:
    lowered = str(name or "").strip().lower()
    special = {
        "__init__": "Initializes the object.",
        "__post_init__": "Normalizes and validates fields after initialization.",
    }
    if lowered in special:
        return special[lowered]
    prefixes = (
        ("build_", "Builds "),
        ("new_", "Creates "),
        ("create_", "Creates "),
        ("make_", "Creates "),
        ("generate_", "Generates "),
        ("collect_", "Collects "),
        ("detect_", "Detects "),
        ("validate_", "Validates "),
        ("assert_", "Asserts "),
        ("encode_", "Encodes "),
        ("decode_", "Decodes "),
        ("stream_", "Streams "),
        ("start_", "Starts "),
        ("stop_", "Stops "),
        ("reset_", "Resets "),
        ("warm_", "Warms "),
        ("prewarm_", "Prewarms "),
        ("enqueue_", "Enqueues "),
        ("submit_", "Submits "),
        ("fetch_", "Fetches "),
        ("read_", "Reads "),
        ("write_", "Writes "),
        ("is_", "Checks whether "),
        ("has_", "Checks whether "),
        ("can_", "Checks whether "),
        ("with_", "Builds "),
        ("to_", "Converts to "),
        ("from_", "Converts from "),
        ("push_", "Pushes "),
        ("pop_", "Pops "),
        ("drain_", "Drains "),
        ("compare_", "Compares "),
        ("plan_", "Plans "),
        ("evaluate_", "Evaluates "),
        ("canonical_", "Canonicalizes "),
        ("remember_", "Stores "),
        ("bind_", "Binds "),
    )
    for prefix, phrase in prefixes:
        if lowered.startswith(prefix):
            target = lowered[len(prefix) :].replace("_", " ").strip() or "data"
            return f"{phrase}{target}."
    return ""


def _humanize_name(name: str) -> str:
    normalized = str(name or "").strip().strip("_")
    if not normalized:
        return "data"
    normalized = re.sub(r"(?<!^)(?=[A-Z])", " ", normalized)
    normalized = normalized.replace("_", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _symbol_summary(name: str, *, kind: str, doc: str) -> str:
    summary = _doc_summary(doc)
    if summary:
        return summary + "."

    phrase = _verb_phrase(name)
    if phrase:
        return phrase

    label = _humanize_name(name)
    if kind == "class":
        return f"Container or runtime type for {label}."
    if kind == "async_method":
        return f"Asynchronously handles {label}."
    if kind == "method":
        return f"Handles {label}."
    if kind == "async_function":
        return f"Asynchronously handles {label}."
    return f"Handles {label}."


def _write_overview(modules: tuple[ModuleInfo, ...]) -> None:
    grouped: dict[str, list[ModuleInfo]] = defaultdict(list)
    for module in modules:
        grouped[_module_title(module.path)].append(module)

    lines: list[str] = []
    lines.append("# Voice Pipeline Python Overview")
    lines.append("")
    lines.append(f"Generated from `apps/api/src/voice_pipeline` on {datetime.now(UTC).date().isoformat()}.")
    lines.append("")
    lines.append("Includes every Python module currently present in the package, with top-level classes, methods, and functions plus short descriptions.")
    lines.append("")
    lines.append(f"Total Python files: {len(modules)}")
    lines.append("")
    for section in sorted(grouped.keys()):
        lines.append(f"## {section}")
        lines.append("")
        for module in sorted(grouped[section], key=lambda item: item.path):
            lines.append(f"### {module.path}")
            lines.append(_module_summary(module))
            lines.append("")
            if module.classes:
                lines.append("### Classes")
                for cls in module.classes:
                    lines.append(f"- {cls.name}: {_symbol_summary(cls.name, kind='class', doc=cls.doc)}")
                    if cls.methods:
                        for method in cls.methods:
                            lines.append(f"  - {cls.name}.{method.name}: {_symbol_summary(method.name, kind=method.kind, doc=method.doc)}")
                    else:
                        lines.append("  - No methods defined in this class.")
            else:
                lines.append("- No top-level classes discovered in this module.")
            lines.append("")
            if module.functions:
                lines.append("### Functions")
                for fn in module.functions:
                    lines.append(f"- {fn.name}: {_symbol_summary(fn.name, kind=fn.kind, doc=fn.doc)}")
            else:
                lines.append("- No top-level functions discovered in this module.")
            lines.append("")
    OVERVIEW_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    modules = _iter_modules()
    _write_symbol_inventory(modules)
    _write_dependency_snapshot(modules)
    _write_overview(modules)
    print(f"generated: {SYMBOL_INVENTORY_PATH}")
    print(f"generated: {DEPENDENCY_SNAPSHOT_PATH}")
    print(f"generated: {OVERVIEW_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
