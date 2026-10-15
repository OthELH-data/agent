"""Filesystem primitives for the markdown precedent wiki.

The directory layout and page format mirror ``or/production-refactor``.  The
OTL sidecar only reads from this module; future build/promote tooling can use
the write helpers without changing the inference contract.
"""
from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path

import yaml


SECTIONS = ("types", "clauses", "precedents", "entities")
_DEFAULT_KB = Path(__file__).resolve().parent / "kb"
_configured_kb = os.environ.get("WIKI_KB_DIR", "").strip()
KB = Path(_configured_kb).expanduser().resolve() if _configured_kb else _DEFAULT_KB
TEMPLATES = Path(__file__).resolve().parent / "templates"


class _FrontMatterLoader(yaml.SafeLoader):
    """Keep date-looking front-matter values as strings."""


_FrontMatterLoader.yaml_implicit_resolvers = {
    key: [item for item in values if item[0] != "tag:yaml.org,2002:timestamp"]
    for key, values in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


def slugify(text: str, maxlen: int = 60) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return value[:maxlen].rstrip("-") or "untitled"


def page_path(section: str, slug: str) -> Path:
    if section not in SECTIONS:
        raise ValueError(f"invalid wiki section: {section!r}")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", slug or ""):
        raise ValueError(f"invalid wiki slug: {slug!r}")
    return KB / section / f"{slug}.md"


def exists(section: str, slug: str) -> bool:
    return page_path(section, slug).is_file()


def read_page(section: str, slug: str) -> str:
    path = page_path(section, slug)
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def list_pages(section: str) -> list[str]:
    if section not in SECTIONS:
        raise ValueError(f"invalid wiki section: {section!r}")
    directory = KB / section
    return sorted(path.stem for path in directory.glob("*.md")) if directory.is_dir() else []


def read_index() -> str:
    path = KB / "index.md"
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def read_template(name: str) -> str:
    path = TEMPLATES / f"{name}.md"
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = handle.name
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def write_page(section: str, slug: str, content: str) -> Path:
    path = page_path(section, slug)
    _atomic_write(path, content)
    return path


def write_index(content: str) -> Path:
    path = KB / "index.md"
    _atomic_write(path, content)
    return path


def append_log(entry: str) -> None:
    path = KB / "log.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(entry.rstrip() + "\n")


def front_matter(md: str) -> dict:
    match = re.match(r"^---\s*\n(.*?)\n---\s*(?:\n|$)", md, re.S)
    if not match:
        return {}
    try:
        payload = yaml.load(match.group(1), Loader=_FrontMatterLoader) or {}
    except yaml.YAMLError:
        return {}
    return payload if isinstance(payload, dict) else {}


def render_front_matter(payload: dict) -> str:
    rendered = yaml.safe_dump(
        payload or {}, allow_unicode=True, default_flow_style=False,
        sort_keys=False, width=1000,
    ).rstrip()
    return f"---\n{rendered}\n---\n"


def body_after_front_matter(md: str) -> str:
    match = re.match(r"^---\s*\n.*?\n---\s*(?:\n|$)", md, re.S)
    return md[match.end():] if match else md


def provenance() -> dict[str, object]:
    """Describe and fingerprint the corpus visible to the current process."""
    digest = hashlib.sha256()
    counts = {section: 0 for section in SECTIONS}
    total_bytes = 0
    file_count = 0
    paths = sorted(KB.rglob("*.md")) if KB.is_dir() else []
    for path in paths:
        if not path.is_file() or path.name == "log.md":
            continue
        relative = path.relative_to(KB).as_posix()
        payload = path.read_bytes()
        digest.update(relative.encode("utf-8") + b"\0" + payload + b"\n")
        total_bytes += len(payload)
        file_count += 1
        section = relative.split("/", 1)[0]
        if section in counts:
            counts[section] += 1
    return {
        "resolved_path": str(KB),
        "selection": "configured_candidate" if _configured_kb else "promoted_default",
        "available": KB.is_dir(),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "section_counts": counts,
        "content_manifest_sha256": digest.hexdigest(),
    }


__all__ = [
    "KB", "TEMPLATES", "SECTIONS", "append_log", "body_after_front_matter",
    "exists", "front_matter", "list_pages", "page_path", "provenance",
    "read_index", "read_page", "read_template", "render_front_matter", "slugify",
    "write_index", "write_page",
]

