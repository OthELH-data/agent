"""Skill registry: deterministic slug match first, LLM family-matching fallback.

A "skill" is a markdown playbook that tells the LLM how to understand a given exception type — e.g.
`skills/skill_material_reconciliation.md`. Resolution is hybrid:

  1. Fast path — the exception type (the CSV `topic`, e.g. "Material Reconciliation") is slugified and
     resolved directly to `skill_<slug>.md`. This is free and deterministic, and also honors an
     explicit `--skill <name>`.
  2. LLM fallback — when no file matches (e.g. the broad topic "Labor Issues", which no single
     `skill_<slug>.md` covers), a light structured pass is shown each candidate skill's
     `**Applies to families:**` header plus the exception description, and picks the one skill whose
     families best fit — or the `no_skill` sentinel, in which case the caller runs the plain pipeline.

Dropping a new `skill_<type>.md` (with an `**Applies to families:**` header) makes that exception type
skill-driven with no code change — the filename and that header are the entire contract.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from extractor.llm_adapter import structured_call
from extractor.schemas import SkillSelection

# skills/ lives next to the package (excel_extractor/skills), i.e. one level above this file's dir.
_DEFAULT_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


@dataclass
class Skill:
    """A resolved skill: its slug name, the markdown body (LLM instructions), and its file path."""
    name: str
    instructions: str
    path: Path


def slugify(exc_type: str) -> str:
    """Normalize an exception type to a filename slug.

    "Material Reconciliation" -> "material_reconciliation"
    "Marine & Air Logistics"  -> "marine_air_logistics"
    "  OTL / Timesheets  "    -> "otl_timesheets"
    """
    s = (exc_type or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)   # any run of non-alphanumerics -> single underscore
    return s.strip("_")


_NO_SKILL = "no_skill"  # sentinel the LLM returns when no candidate skill applies


def _extract_families_header(text: str) -> str:
    """The routing-relevant top of a skill file: its H1 title + the `**Applies to families:**` block.

    Returns the first `# ` H1 line plus, if present, the `**Applies to families:**` label line and its
    wrapped continuation lines. The family list wraps across lines that each begin with a backtick, so
    continuation stops at the first line that is not such a wrap (blank line, or the next section —
    which may be `**bold**`, a `>` blockquote, `##`, `---`, etc.). Falls back to just the title when no
    families label exists (e.g. an older skill that leads with a different `**` section).
    """
    lines = text.splitlines()
    title = next((ln.strip() for ln in lines if ln.lstrip().startswith("# ")), "")
    families: list[str] = []
    for i, ln in enumerate(lines):
        if ln.startswith("**Applies to families:**"):
            families.append(ln.strip())
            for cont in lines[i + 1:]:            # family list wraps onto backtick-led lines only
                if not cont.strip().startswith("`"):
                    break
                families.append(cont.strip())
            break
    parts = [p for p in (title, "\n".join(families)) if p]
    return "\n".join(parts)


def _available_skills(skills_dir: str | Path | None = None) -> dict[str, Skill]:
    """Enumerate `skill_<slug>.md` files in the skills dir → {slug: Skill}."""
    base = Path(skills_dir) if skills_dir else _DEFAULT_SKILLS_DIR
    out: dict[str, Skill] = {}
    for path in sorted(base.glob("skill_*.md")):
        slug = path.stem[len("skill_"):]
        if slug:
            out[slug] = Skill(name=slug, instructions=path.read_text(encoding="utf-8"), path=path)
    return out


_SYSTEM_SKILL_SELECT = (
    "You route an auditor's exception to the single best domain skill. You are given the exception's "
    "topic and description, and a list of candidate skills — each with its slug and its "
    "'Applies to families' header (the exception families the skill covers). The auditor's free-text "
    "title/topic is NOT reliable routing authority; go by what the description actually describes and "
    "match it to the families. Return exactly ONE slug copied verbatim from the candidate list, or the "
    "literal 'no_skill' if none of the skills' families fit the exception."
)


def _llm_select_skill(exc_type: str, description: str, skills: dict[str, Skill]) -> Skill | None:
    """Light structured pass: pick one candidate skill by its families header, or None (no_skill)."""
    catalog = "\n\n".join(
        f"slug: {slug}\n{_extract_families_header(sk.instructions)}" for slug, sk in skills.items()
    )
    user = (
        f"EXCEPTION TOPIC: {exc_type}\n\n"
        f"EXCEPTION DESCRIPTION:\n{description or '(none provided)'}\n\n"
        f"CANDIDATE SKILLS:\n{catalog}\n\n"
        f"Return one slug from the candidates, or '{_NO_SKILL}' if none apply."
    )
    sel = structured_call("light", _SYSTEM_SKILL_SELECT, user, SkillSelection)
    slug = slugify(sel.skill_slug)
    return skills.get(slug)  # None for no_skill / any unrecognized value → plain pipeline


def resolve_skill(exc_type: str, description: str = "", skills_dir: str | Path | None = None,
                  use_llm: bool = True) -> Skill | None:
    """Resolve an exception to a Skill, or None to fall back to the plain pipeline.

    Hybrid resolution: first try the deterministic slug match (`skill_<slug>.md`, which also honors an
    explicit `--skill` name); if none matches and `use_llm` is set, ask a light structured pass to pick
    the best skill by its `**Applies to families:**` header against the exception `description`. Returns
    None when no file matches AND the LLM reports `no_skill` (or LLM routing is disabled).
    """
    slug = slugify(exc_type)
    base = Path(skills_dir) if skills_dir else _DEFAULT_SKILLS_DIR
    # 1. Fast path: direct slug match (tolerate an explicit name already carrying the skill_ prefix).
    if slug:
        direct = slug[len("skill_"):] if slug.startswith("skill_") else slug
        path = base / f"skill_{direct}.md"
        if path.is_file():
            return Skill(name=direct, instructions=path.read_text(encoding="utf-8"), path=path)

    # 2. LLM fallback: let the model match the description to a skill's families header.
    if not use_llm:
        return None
    skills = _available_skills(skills_dir)
    if not skills:
        return None
    return _llm_select_skill(exc_type, description, skills)
