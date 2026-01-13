"""Utility helpers used by agent-level orchestrators."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Mock utils for analysis if they don't exist
try:
    from utils import get_workspace_paths, load_mapping
except ImportError:
    # Fallback/Mock for analysis purposes
    def get_workspace_paths():
        return {
            "script_root": "env_scripts",
            "mapping_file": "mapping.txt",
            "script_extensions": [".nvn"]
        }
    def load_mapping(path):
        return {}

PLACEHOLDER_RE = re.compile(r"\\?\{([A-Za-z0-9_]+)\}")
CONST_DECL_RE = re.compile(r'^\s*const\s+([A-Za-z0-9_]+)\s*=\s*"(.*)"\s*$')
STRING_LITERAL_RE = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"')


def scan_script_references(
    target_path: str,
    *,
    keyword: Optional[str] = None,
    scripts_root: Path | str | None = None,
    mapping_path: Path | str | None = None,
    script_extensions: Optional[Sequence[str]] = None,
    emit: bool = True,
) -> Dict[str, Any]:
    """Locate references to ``target_path`` in mirrored scripts.

    When ``keyword`` is ``"read"`` or ``"write"``, only the corresponding DSL statements
    are inspected. If ``keyword`` is omitted, both verbs are considered and the function also
    falls back to matching any quoted literal that resolves to the requested path.
    
    Special mode: If ``target_path`` is "*" or empty, lists ALL statements matching the keyword
    (useful for listing all imports, all reads, etc.)
    """

    normalized_target = (target_path or "").strip()
    
    # Special mode: list all statements of given keyword type
    list_all_mode = normalized_target in ("*", "")
    
    if not list_all_mode:
        target_variants = {normalized_target}
        stripped = normalized_target.lstrip("/")
        if stripped and stripped != normalized_target:
            target_variants.add(stripped)
        if not normalized_target.startswith("/"):
            target_variants.add(f"/{normalized_target}")
    else:
        target_variants = set()  # Match everything in list_all_mode
        
    defaults = get_workspace_paths()
    scripts_dir = Path(scripts_root or defaults["script_root"])
    if not scripts_dir.exists():
        raise FileNotFoundError(f"Scripts directory not found: {scripts_dir}")
    mapping_source = mapping_path or defaults["mapping_file"]
    mapping = load_mapping(mapping_source)
    search_literals = keyword is None and not list_all_mode
    verbs = [keyword.lower()] if keyword else ["read", "write", "import"]
    verbs_pattern = "|".join(re.escape(v) for v in verbs)
    # Accept both `write "/..."` and `write:"/..."` syntax emitted by NVN scripts.
    statement_re = re.compile(rf"\b({verbs_pattern})\s*:?\s*\"(.*?)\"", re.IGNORECASE)
    findings: List[Dict[str, Any]] = []
    total_occurrences = 0

    extensions = _normalize_extensions(script_extensions or defaults["script_extensions"])
    script_paths: List[Path] = []
    seen: set[Path] = set()
    for ext in extensions:
        pattern = f"*{ext}"
        for candidate in scripts_dir.glob(pattern):
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            script_paths.append(candidate)
    script_paths.sort()

    for script_path in script_paths:
        consts = _collect_constants(script_path)
        hits: List[Dict[str, Any]] = []
        lines = script_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line_no, raw_line in enumerate(lines, start=1):
            recorded = False
            match = statement_re.search(raw_line)
            if match:
                verb = match.group(1).lower()
                literal = match.group(2)
                resolved = _resolve_placeholders(literal, consts)
                normalized = resolved.replace("\\", "")
                # In list_all_mode, accept all matches; otherwise check target_variants
                if list_all_mode or any(variant in normalized for variant in target_variants):
                    hits.append(
                        {
                            "line": line_no,
                            "verb": verb,
                            "raw": raw_line.strip(),
                            "resolved_path": normalized,
                        }
                    )
                    recorded = True
            if search_literals:
                for literal in STRING_LITERAL_RE.findall(raw_line):
                    resolved_literal = _resolve_placeholders(literal, consts)
                    normalized_literal = resolved_literal.replace("\\", "")
                    if not any(variant in normalized_literal for variant in target_variants):
                        continue
                    if recorded and any(
                        existing["line"] == line_no and existing["resolved_path"] == normalized_literal
                        for existing in hits
                    ):
                        continue
                    hits.append(
                        {
                            "line": line_no,
                            "verb": "literal",
                            "raw": raw_line.strip(),
                            "resolved_path": normalized_literal,
                        }
                    )
        if hits:
            script_id = script_path.stem
            findings.append(
                {
                    "script_id": script_id,
                    "file_path": mapping.get(script_id, "(unknown path)"),
                    "hits": hits,
                }
            )
            total_occurrences += len(hits)

    payload = {
        "target": normalized_target if normalized_target else "*",
        "mode": "list_all" if list_all_mode else "search",
        "keyword": keyword or "any",
        "scripts_count": len(findings),
        "occurrences": total_occurrences,
        "results": findings,
    }
    if emit:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def _collect_constants(script_path: Path) -> Dict[str, str]:
    consts: Dict[str, str] = {}
    lines = script_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for raw in lines:
        match = CONST_DECL_RE.match(raw.strip())
        if not match:
            continue
        key, value = match.group(1), match.group(2)
        consts[key] = _resolve_placeholders(value, consts)
    return consts


def _resolve_placeholders(text: str, consts: Dict[str, str], *, depth: int = 0) -> str:
    if depth > 10:
        return text
    replaced = PLACEHOLDER_RE.sub(lambda match: consts.get(match.group(1), ""), text)
    if replaced == text:
        return replaced
    return _resolve_placeholders(replaced, consts, depth=depth + 1)


def _normalize_extensions(exts: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    for ext in exts:
        if not ext:
            continue
        cleaned = str(ext).strip()
        if not cleaned:
            continue
        if not cleaned.startswith("."):
            cleaned = f".{cleaned}"
        normalized.append(cleaned)
    deduped = list(dict.fromkeys(normalized))
    return deduped or [".nvn"]
