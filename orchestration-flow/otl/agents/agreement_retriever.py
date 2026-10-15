"""Asset-scoped agreement retrieval immediately before the drafter.

The auditor's original exception text is the primary query.  Specialist findings form a separate,
additive Shell-theory query so a defensible route that the auditor did not raise can still retrieve its
governing language.  LLM calls return candidate IDs and labels only; source text is reloaded from
SQLite after validation and reaches the drafter verbatim.

One pipeline serves both retrieval modes (``OTL_AGREEMENT_RETRIEVAL_MODE``, default ``bm25``); the only
difference is whether vector search is attempted:

  1. Searches: the main searches are the auditor query (perspective ``auditor_theory``) and, when
     non-empty and different, the Shell findings query (perspective ``shell_findings``). Ground searches:
     every ``_GROUND_SPECS`` legal ground whose patterns occur in a main query adds one search (deduped,
     in ``_GROUND_SPECS`` order) whose text is the ground's expansion string, source
     ``ground:<ground>``, perspective ``auditor_theory`` if the auditor query fired it else
     ``shell_findings`` (its rows carry both when both fired it).
  2. Query vectors (hybrid only): a compatibility check against the index, then ONE embedding call for
     every distinct search text, main and ground (``search_texts``). Any failure records
     ``fallback_reason`` and every search, ground searches included, continues without vectors.
  3. Per search (``_retrieve_perspective``, main and ground alike): a flat BM25 list over all applicable
     documents and, when the search has a vector, a flat cosine-KNN list. Both non-empty -> RRF (k=60);
     exactly one non-empty -> that list in its own retriever order (direct ranking, no RRF); both
     empty -> nothing.
  4. Pool (``_merge_candidates`` + ``_assemble_pool``): the main searches only are merged (two non-empty
     lists RRF-fused, perspectives unioned; a single list used directly); ground searches never enter
     that RRF. Each ground search reserves its top ``_GROUND_RESERVE`` rows (deduped across grounds,
     first ground wins). With pool size P (``OTL_AGREEMENT_POOL_SIZE``, default ``MAX_CANDIDATES``),
     m is the largest count with m + |reserved not in main[:m]| <= P; the pool is main[:m] followed
     by those reserved chunks (reserve AFTER main). A reserved chunk already in main[:m] keeps its
     place; every reserved row is tagged ``retrieval_grounds``. No ground fired -> main[:P], as before.
  5. One ``agreement_retriever`` listwise rerank labels the pool governing | supporting | auditor_cited |
     not_relevant.
  6. ``_select_ranked`` takes the top ``TOP_K`` by label then reranker
     order, skipping near duplicates, with no per-track or per-perspective quota.
  7. The selected IDs are re-read from SQLite; that text, not the pool rows or the LLM output, is final.

Steps 5-6 are ``rank_pool`` (also used by ``otl.rag_tests.replay_rerank``); steps 3-4 are
``candidate_pool`` (also used by ``otl.rag_tests.funnel`` and ``otl.rag_tests.retrieve_candidates``).
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, Field

from otl import llm_adapter as llm
from otl.chargeability_hooks import evaluate_state
from otl.agreements.corpus import (
    canonical_asset, date_filter_provenance, documents_for_case, load_manifest,
)
from otl.agreements.embedding import (
    AzureEmbedder,
    EmbeddingConfigError,
    EmbeddingSettings,
    RUNTIME_MAX_RETRIES,
    RUNTIME_TIMEOUT_SECONDS,
    bound_embedding_input,
    validate_vectors,
)
from otl.agreements.fusion import RRF_K, rrf_lists
from otl.agreements.store import AgreementIndex
from otl.state import AgreementChunk, AgreementRetrievalState
from otl.trace import trace_event
from otl.agents._common import active_auditor_text, people_coverage, rebuttal_context_block


# Default number of chunks kept after the final rerank.
TOP_K = 5
# Default candidate-pool size sent to the reranker (overridable by OTL_AGREEMENT_POOL_SIZE).
MAX_CANDIDATES = 40
# Depth of every BM25 and vector list.
_SEARCH_DEPTH = 100
# Top rows of each ground search reserved in the pool (after the main chunks).
_GROUND_RESERVE = 2
_MAX_QUERY_CHARS = 20000
_CANDIDATE_EXCERPT_CHARS = 1200

_GROUND_SPECS: OrderedDict[str, tuple[tuple[str, ...], str]] = OrderedDict({
    "regulatory_activity": (
        ("regulatory", "30 cfr", "bsee", "sems", "apd", "blowout preventer", "bop"),
        "regulatory activity BSEE SEMS law regulation drilling completion well intervention",
    ),
    "feasibility_project_studies": (
        ("feasibility", "project studies", "field improvement", "facility improvement"),
        "feasibility project studies field facility improvements approved AFE",
    ),
    "planning_maintenance_turnaround": (
        ("planner", "scheduler", "turnaround", "maintenance", "rig administration"),
        "planners schedulers maintenance turnaround rig administration relocated onshore",
    ),
    "technical_employee_exclusion": (
        ("excluded", "administration", "secretarial", "management", "overhead", "not technical"),
        "Technical Employee exclusions administration management supervision overhead",
    ),
    "technical_employee": (
        ("technical employee", "technical/engineering", "technical in nature", "engineering in nature",
         "operational skill", "special and specific"),
        "offsite technical labor Technical Employees engineering geological geophysical technical "
        "operational skills direct charge governing agreement",
    ),
    "offsite_technical_labor": (
        ("offsite technical labor", "offsite labor", "technical labor", "technical employee",
         "technical in nature", "off the joint property"),
        "offsite technical labor technical employees role responsibilities location Joint Property "
        "principal benefit direct charge overhead governing contract accounting procedure",
    ),
    "affiliate_labor": (
        ("affiliate labor", "affiliate otl", "affiliate-performed", "affiliate"),
        "affiliate labor affiliate offsite technical labor same chargeability standards",
    ),
    "major_construction": (
        ("major construction", "mcoh", "design construction installation"),
        "major construction design construction installation direct charge regardless location",
    ),
    "major_construction_timewriting_threshold": (
        ("timewriting", "time writing", "monthly hours", "eight hours", "8 hours", "cats labor"),
        "major construction employee timewriting monthly minimum hours threshold personnel "
        "applicability exceptions direct charge",
    ),
    "location_joint_property_benefit": (
        ("offsite", "joint property", "physical location", "principal benefit", "principle benefit"),
        "offsite Joint Property principal benefit Joint Operations physical location",
    ),
    "supervision": (
        ("first-level supervision", "first level supervision", "above first level", "reporting level"),
        "first level supervision above first level management overhead chargeability",
    ),
})

_SENTENCE_ANCHORS = tuple(
    term for patterns, _expansion in _GROUND_SPECS.values() for term in patterns
) + ("chargeable", "non-chargeable", "not chargeable", "unresolved", "mixed population")

_PERSPECTIVES = ("auditor_theory", "shell_findings")
_RELATIONSHIPS = ("supports", "limits", "contradicts", "context")
# Selection order of the reranker labels; not_relevant is never selected.
_LABEL_ORDER = {"governing": 0, "supporting": 1, "auditor_cited": 2}
_LABEL_ALIASES = {"direct": "governing"}


class _RankedCandidate(BaseModel):
    chunk_id: str
    relevance: str = Field(description="governing | supporting | auditor_cited | not_relevant")
    perspectives: list[str] = Field(
        default_factory=list, description="auditor_theory and/or shell_findings")
    relationship: str = Field(default="context", description="supports | limits | contradicts | context")
    selection_reason: str = ""


class _AgreementRanking(BaseModel):
    ranked_candidates: list[_RankedCandidate] = Field(default_factory=list)


_RERANK_SYSTEM = """You rank exact excerpts from legal agreements for a joint-venture OTL audit response.

Return candidate IDs and labels only; never reproduce, edit, summarize, or invent agreement language. Read the \
AUDITOR THEORY and any ADDITIVE SHELL FINDINGS, then decide which clause actually resolves the dispute. The \
auditor usually relies on the operating agreement's accounting procedure; the deciding clause is often other \
authority between the parties (a settlement agreement, billing standard, attestation, or amendment) that carves \
the cost category out of that procedure, redefines it, or overrides it. Label each candidate you return:
- governing: the excerpt states the rule that decides whether a disputed cost category is chargeable for these \
parties. A settlement, billing standard, or amendment provision that overrides or carves out the general \
accounting procedure governs over that procedure. When nothing overrides the auditor's authority, that \
authority itself is governing.
- supporting: definitions, scope, documentation, or procedure language a response needs alongside a governing \
clause.
- auditor_cited: authority the auditor relies on that a governing clause overrides or narrows; context, not \
the deciding rule.
- not_relevant: no material bearing on the dispute.
Several clauses from the same document may all be governing; never drop one to represent another document. \
Prefer one clause over a near-identical copy. Diversity across documents or tracks only breaks ties between \
equally strong candidates.

Use only IDs present in CANDIDATES. perspectives may contain only auditor_theory and shell_findings \
(shell_findings means the clause serves a route raised by the Shell findings rather than the auditor). \
relationship states how the clause bears on the auditor's theory: supports, limits, contradicts, or context. \
Rank strongest first."""


def _dump(value: Any) -> dict:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return dict(value) if isinstance(value, dict) else {}


def _bounded(text: str, maximum: int = _MAX_QUERY_CHARS) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= maximum:
        return value
    head = int(maximum * 0.7)
    return value[:head].rstrip() + " … " + value[-(maximum - head - 3):].lstrip()


def auditor_query(state: dict) -> str:
    intake = _dump(state.get("intake"))
    case = state.get("case") or {}
    original = (
        active_auditor_text(state)
        or state.get("rebuttal_current_posture")
        or intake.get("nl_summary")
        or case.get("description")
        or case.get("desc")
        or case.get("title")
        or ""
    )
    return _bounded(str(original))


def _salient_sentences(text: str, maximum: int = 1_200) -> str:
    """Choose chargeability-bearing sentences before applying a cap; never prefix-truncate blindly."""
    sentences = re.split(r"(?<=[.!?])\s+|\n+", str(text or ""))
    selected: list[str] = []
    used = 0
    for sentence in sentences:
        clean = " ".join(sentence.split())
        lower = clean.casefold()
        if not clean or not any(anchor in lower for anchor in _SENTENCE_ANCHORS):
            continue
        if selected and used + len(clean) + 1 > maximum:
            break
        selected.append(clean)
        used += len(clean) + 1
    return " ".join(selected)


def shell_findings_query(state: dict) -> str:
    """Build an additive, row-free Shell theory without another summarization model call."""
    jade = _dump(state.get("jade_analyzer"))
    location = _dump(state.get("location"))
    job = _dump(state.get("job_title"))
    matching = _dump(state.get("matching"))
    supervision = _dump(state.get("supervision"))
    if not any((jade, location, job, matching, supervision)):
        return ""

    source_by_profile: dict[str, str] = {}
    for profile, rationale in (matching.get("cfr_rationale") or {}).items():
        source_by_profile[str(profile)] = str(rationale or "")
    for profile, summary in (job.get("identified_job_data") or {}).items():
        source_by_profile.setdefault(str(profile), str(summary or ""))

    groups: defaultdict[str, list[str]] = defaultdict(list)
    combined = "\n".join([
        str(matching.get("nl_summary") or ""),
        str(job.get("technical_rationale") or ""),
        str(location.get("nl_summary") or ""),
        str(supervision.get("nl_summary") or ""),
    ]).casefold()
    for ground, (patterns, _expansion) in _GROUND_SPECS.items():
        for profile, rationale in source_by_profile.items():
            haystack = f"{profile} {rationale}".casefold()
            if any(pattern in haystack for pattern in patterns):
                groups[ground].append(profile)
        if not groups[ground] and any(pattern in combined for pattern in patterns):
            groups[ground] = []

    if jade.get("major_construction_charge"):
        groups["major_construction"] = []
    monthly = jade.get("mcoh_monthly_facts") or []
    if monthly:
        groups["major_construction_timewriting_threshold"] = []
    if location:
        groups["location_joint_property_benefit"] = []
    if supervision:
        groups["supervision"] = []
    if (job.get("technical_job_title") is True
            or any(value == "technical" for value in (job.get("role_status_by_person") or {}).values())):
        groups.setdefault("technical_employee", [])
        groups.setdefault("offsite_technical_labor", [])
    if any(value == "matched" for value in
           (matching.get("cfr_mapping_status_by_person") or {}).values()):
        groups.setdefault("offsite_technical_labor", [])

    lines: list[str] = []
    people, dropped, total = people_coverage(state)
    if people or dropped:
        lines.append(
            f"Population coverage: {len(people)} of {total} named people were examined; "
            f"{len(dropped)} remain outside the specialist findings."
        )
    for ground, (_patterns, expansion) in _GROUND_SPECS.items():
        if ground not in groups:
            continue
        profiles = list(dict.fromkeys(groups[ground]))[:8]
        detail = f" Representative profiles: {', '.join(profiles)}." if profiles else ""
        lines.append(f"Legal ground - {ground.replace('_', ' ')}: {expansion}.{detail}")
    if monthly:
        hours = [float(item.get("hours") or 0.0) for item in monthly if isinstance(item, dict)]
        below = sum(bool(item.get("below_threshold")) for item in monthly if isinstance(item, dict))
        if hours:
            lines.append(
                "MCOH timewriting threshold question: "
                f"{len(hours)} identified person-month group(s), submitted-hour range "
                f"{min(hours):g}-{max(hours):g}, {below} below the provisional 8-hour threshold. "
                "Retrieve the governing monthly threshold, covered personnel, timing conditions, "
                "exceptions, and chargeability consequence."
            )

    # Matching/job-title summaries can add category language omitted by older typed outputs. Location
    # and supervision summaries commonly contain person names, which have no agreement-retrieval value.
    for stage in (matching, job):
        salient = _salient_sentences(stage.get("nl_summary") or "", maximum=2_000)
        if salient:
            lines.append(salient)
    mapped_terms = list(dict.fromkeys(
        value for mapping in (
            matching.get("cfr_primary_by_person") or {},
            matching.get("cfr_secondary_by_person") or {},
            matching.get("cfr_additional_by_person") or {},
            matching.get("subpart_labels_by_person") or {},
        ) for values in mapping.values() for value in (values or []) if str(value).strip()
    ))
    if mapped_terms:
        lines.append(
            "Offsite technical labor regulatory context: " + "; ".join(mapped_terms[:20])
            + ". Retrieve the asset-specific governing contract provisions that determine whether the "
              "documented roles and responsibilities are directly chargeable, excluded, or covered by overhead."
        )
    return _bounded("\n".join(lines), 8000)



def _candidate_excerpt(text: str, limit: int = _CANDIDATE_EXCERPT_CHARS) -> str:
    """A bounded prompt excerpt: the head and tail of the chunk text, never the full chunk."""
    clean = str(text or "").strip()
    if len(clean) <= limit:
        return clean
    head = int(limit * 0.8)
    return clean[:head].rstrip() + "\n…\n" + clean[-(limit - head - 3):].lstrip()


# --- Retrieval mode -------------------------------------------------------------------------------

class _Fallback(Exception):
    """Raised by query-vector preparation: the run continues without vectors, with this reason."""

    def __init__(self, reason: str, error_type: str | None = None):
        super().__init__(reason)
        self.reason = reason
        self.error_type = error_type


def _resolve_mode() -> tuple[str, str | None, str | None, str | None]:
    """Read and normalize OTL_AGREEMENT_RETRIEVAL_MODE; returns (requested, effective, reason, error_type).

    `effective` is None for a hybrid request until the searches have run, because only then is it known
    whether vectors contributed. A bm25 request (by default, explicit value, or invalid value) is decided
    immediately: no vector search is attempted.
    """
    raw = (os.getenv("OTL_AGREEMENT_RETRIEVAL_MODE") or "").strip().lower()
    if raw == "hybrid":
        return "hybrid", None, None, None
    reason = "invalid_retrieval_mode" if raw not in ("", "bm25") else None
    return "bm25_only", "bm25_only", reason, None


def _mode_fields(
    requested_mode: str, effective_mode: str | None, fallback_reason: str | None,
    fallback_error_type: str | None,
) -> dict[str, Any]:
    return {
        "requested_mode": requested_mode,
        "effective_mode": effective_mode,
        "fallback_reason": fallback_reason,
        "fallback_error_type": fallback_error_type,
    }


def _effective_mode(requested_mode: str, per_search: list[dict[str, Any]]) -> str:
    """hybrid if any search fused lexical+vector; vector_only if vectors alone fed every non-empty search."""
    if requested_mode != "hybrid":
        return "bm25_only"
    rankings = [search["ranking"] for search in per_search if search["ranking"] != "empty"]
    if "rrf" in rankings:
        return "hybrid"
    if rankings and all(ranking == "vector_direct" for ranking in rankings):
        return "vector_only"
    return "bm25_only"


def top_k() -> int:
    """Final selection size after the rerank (default ``TOP_K``)."""
    return TOP_K


def _max_candidates() -> int:
    """Candidate-pool size sent to the reranker (``OTL_AGREEMENT_POOL_SIZE``, default ``MAX_CANDIDATES``)."""
    try:
        return max(1, int(os.getenv("OTL_AGREEMENT_POOL_SIZE", MAX_CANDIDATES)))
    except ValueError:
        return MAX_CANDIDATES


# --- Query vectors (hybrid only) ------------------------------------------------------------------

@dataclass
class _SemanticPrep:
    """Query vectors from the single embedding call, plus what provenance reports about it."""

    vectors_by_text: dict[str, list[float]]
    model: str
    dimensions: int
    input_count: int
    input_shortened: bool
    embedding_seconds: float

    def provenance(self) -> dict[str, Any]:
        return {
            "embedding_model": self.model,
            "embedding_dimensions": self.dimensions,
            "embedding_input_count": self.input_count,
            "embedding_input_shortened": self.input_shortened,
            "embedding_seconds": round(self.embedding_seconds, 3),
        }


def _check_semantic_compatibility(index: AgreementIndex) -> EmbeddingSettings:
    """Confirm the index carries vectors matching the configured deployment, without any Azure call."""
    if not index.has_vectors():
        raise _Fallback("index_has_no_vectors")
    try:
        settings = EmbeddingSettings.from_env()
    except EmbeddingConfigError:
        raise _Fallback("embedding_not_configured") from None
    metadata = index.metadata()
    if settings.model != metadata.get("embedding_model"):
        raise _Fallback("embedding_model_mismatch")
    if settings.dimensions != int(metadata.get("embedding_dimensions") or 0):
        raise _Fallback("embedding_dimensions_mismatch")
    return settings


def _embed_queries(settings: EmbeddingSettings, query_texts: list[str]) -> _SemanticPrep:
    """Embed every query text in ONE request; any failure is a sanitized `_Fallback`."""
    try:
        shortened = False
        bounded_texts: list[str] = []
        for text in query_texts:
            bounded, was_shortened = bound_embedding_input(text)
            bounded_texts.append(bounded)
            shortened = shortened or was_shortened
        embedder = AzureEmbedder(
            settings, max_retries=RUNTIME_MAX_RETRIES, timeout=RUNTIME_TIMEOUT_SECONDS,
        )
        start = time.perf_counter()
        vectors = embedder.embed(bounded_texts)
        vectors = validate_vectors(vectors, len(bounded_texts), settings.dimensions)
        embedding_seconds = time.perf_counter() - start
    except Exception as exc:
        raise _Fallback("embedding_request_failed", type(exc).__name__) from None
    return _SemanticPrep(
        vectors_by_text=dict(zip(query_texts, vectors)), model=settings.model,
        dimensions=settings.dimensions, input_count=len(query_texts), input_shortened=shortened,
        embedding_seconds=embedding_seconds,
    )


# --- Candidate pool: per-search ranking, then cross-search merge ----------------------------------

@dataclass(frozen=True)
class _Search:
    text: str
    source: str  # "auditor_query" | "shell_findings" | "ground:<ground>"
    perspective: str  # "auditor_theory" | "shell_findings"
    # Perspectives stamped on this search's rows; empty means (perspective,). A ground search fired by
    # both main queries carries both.
    perspectives: tuple[str, ...] = ()

    def row_perspectives(self) -> list[str]:
        return list(self.perspectives or (self.perspective,))


def _searches(auditor: str, shell: str) -> list[_Search]:
    """The auditor query, plus the Shell findings query when non-empty and different."""
    searches: list[_Search] = []
    if auditor:
        searches.append(_Search(text=auditor, source="auditor_query", perspective="auditor_theory"))
    if shell and shell != auditor:
        searches.append(_Search(text=shell, source="shell_findings", perspective="shell_findings"))
    return searches


def _detected_ground_queries(query: str) -> list[tuple[str, str]]:
    """``(ground, expansion)`` for every ``_GROUND_SPECS`` entry whose patterns occur in the query."""
    lower = str(query or "").casefold()
    return [
        (ground, expansion)
        for ground, (patterns, expansion) in _GROUND_SPECS.items()
        if any(pattern in lower for pattern in patterns)
    ]


def _ground_searches(main: Sequence[_Search]) -> list[_Search]:
    """One search per legal ground detected in any main search, in ``_GROUND_SPECS`` order.

    The search text is the ground's expansion string. Its perspective is ``auditor_theory`` when the
    auditor query fired it, else ``shell_findings``; its rows carry every perspective that fired it.
    """
    fired: dict[str, set[str]] = defaultdict(set)
    for search in main:
        for ground, _expansion in _detected_ground_queries(search.text):
            fired[ground].add(search.perspective)
    grounds: list[_Search] = []
    for ground, (_patterns, expansion) in _GROUND_SPECS.items():
        if ground not in fired:
            continue
        perspectives = tuple(value for value in _PERSPECTIVES if value in fired[ground])
        grounds.append(_Search(
            text=expansion, source=f"ground:{ground}", perspective=perspectives[0],
            perspectives=perspectives,
        ))
    return grounds


def _all_searches(auditor: str, shell: str) -> tuple[list[_Search], list[_Search]]:
    main = _searches(auditor, shell)
    return main, _ground_searches(main)


def search_texts(auditor: str, shell: str = "") -> list[str]:
    """Every distinct search text (main searches, then ground searches): the single embedding input."""
    main, grounds = _all_searches(auditor, shell)
    return list(dict.fromkeys(search.text for search in (*main, *grounds)))


@dataclass
class _SearchResult:
    """One search's ranked rows and how they were ranked."""

    perspective: str
    rows: list[dict[str, Any]]
    lexical_count: int
    vector_count: int
    ranking: str  # "rrf" | "lexical_direct" | "vector_direct" | "empty"
    vector_error_type: str | None = None
    source: str = ""

    def provenance(self) -> dict[str, Any]:
        return {
            "perspective": self.perspective, "source": self.source,
            "lexical_count": self.lexical_count, "vector_count": self.vector_count,
            "ranking": self.ranking,
        }


def _retrieve_perspective(
    index: AgreementIndex, search: _Search, documents: list[Any],
    vector: Sequence[float] | None = None, depth: int = _SEARCH_DEPTH,
) -> _SearchResult:
    """BM25 (and, with a vector, cosine-KNN) over all applicable documents, ranked for one search.

    Both lists non-empty -> RRF; exactly one non-empty -> that list in its own retriever order;
    both empty -> no rows. A vector-search failure is an empty vector list plus its error type; a
    lexical failure propagates.
    """
    lexical = index.lexical_ranking(search.text, documents, depth=depth)
    vector_rows: list[dict[str, Any]] = []
    error_type: str | None = None
    if vector is not None:
        try:
            vector_rows = index.vector_ranking(vector, documents, depth=depth)
        except Exception as exc:
            vector_rows, error_type = [], type(exc).__name__

    rows: list[dict[str, Any]] = []
    if lexical and vector_rows:
        ranking = "rrf"
        for row in rrf_lists({"lexical": lexical, "vector": vector_rows}, k=RRF_K):
            lists = {entry["list"] for entry in row.pop("retrieval_lists")}
            row.pop("fusion_score", None)
            row.pop("fusion_rank", None)
            row["retrieval_methods"] = [method for method in ("lexical", "vector") if method in lists]
            row["perspectives"] = search.row_perspectives()
            rows.append(row)
    elif lexical or vector_rows:
        method = "lexical" if lexical else "vector"
        ranking = f"{method}_direct"
        rows = [dict(row, retrieval_methods=[method], perspectives=search.row_perspectives())
                for row in (lexical or vector_rows)]
    else:
        ranking = "empty"
    return _SearchResult(
        perspective=search.perspective, rows=rows, lexical_count=len(lexical),
        vector_count=len(vector_rows), ranking=ranking, vector_error_type=error_type,
        source=search.source,
    )


def _merge_candidates(
    per_search: Sequence[list[dict[str, Any]]], limit: int | None,
) -> tuple[list[dict[str, Any]], str]:
    """Merge per-search lists into one pool of at most ``limit`` rows; returns (pool, ranking).

    Two or more non-empty lists are RRF-fused by their positions (ranking "rrf"; perspectives and
    retrieval methods are unioned for a chunk several searches found); a single list is used as is
    (ranking "direct"). Each pool row gets ``fusion_rank``: its 1-based position in the pool.
    ``limit=None`` keeps every merged row.
    """
    non_empty = [rows for rows in per_search if rows]
    if len(non_empty) >= 2:
        ranking = "rrf"
        named = {f"search_{position}": rows for position, rows in enumerate(non_empty)}
        by_list = {name: {row["chunk_id"]: row for row in rows} for name, rows in named.items()}
        merged: list[dict[str, Any]] = []
        for row in rrf_lists(named, k=RRF_K):
            perspectives: list[str] = []
            methods: list[str] = []
            for entry in row.pop("retrieval_lists"):
                source = by_list[entry["list"]][row["chunk_id"]]
                perspectives.extend(source.get("perspectives") or [])
                methods.extend(source.get("retrieval_methods") or [])
            row.pop("fusion_score", None)
            row["perspectives"] = list(dict.fromkeys(perspectives))
            row["retrieval_methods"] = [method for method in ("lexical", "vector") if method in methods]
            merged.append(row)
    else:
        ranking = "direct"
        merged = [dict(row) for row in (non_empty[0] if non_empty else [])]
    if limit is not None:
        merged = merged[:limit]
    for rank, row in enumerate(merged, start=1):
        row["fusion_rank"] = rank
    return merged, ranking


def _union_ground(row: dict[str, Any], ground: str, ground_row: dict[str, Any]) -> None:
    """Tag ``row`` with a ground reserve: add the ground and union its perspectives and methods."""
    row["retrieval_grounds"] = list(dict.fromkeys([*(row.get("retrieval_grounds") or []), ground]))
    row["perspectives"] = list(dict.fromkeys(
        [*(row.get("perspectives") or []), *(ground_row.get("perspectives") or [])]))
    methods = {*(row.get("retrieval_methods") or []), *(ground_row.get("retrieval_methods") or [])}
    row["retrieval_methods"] = [method for method in ("lexical", "vector") if method in methods]


def _assemble_pool(
    main: list[dict[str, Any]], grounds: Sequence[tuple[str, list[dict[str, Any]]]], pool_size: int,
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
    """Main pool plus each ground's top ``_GROUND_RESERVE`` rows, reserved AFTER the main chunks.

    ``main`` is the uncut cross-search merge; ``grounds`` is ``[(ground, rows)]`` in ``_GROUND_SPECS``
    order. Reserved chunks are deduplicated across grounds (first ground wins). ``m`` is the largest
    count with ``m + |reserved not in main[:m]| <= pool_size``; the pool is ``main[:m]`` followed by the
    reserved chunks not already in it, in reserved order. A reserved chunk inside ``main[:m]`` keeps its
    main position. Returns (pool with 1-based ``fusion_rank``, m, per-ground provenance).
    """
    reserved: list[tuple[str, dict[str, Any]]] = []  # (owning ground, row), first ground wins
    tags: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)  # every ground reserving it
    reserved_ids: list[list[str]] = []
    for ground, rows in grounds:
        ids: list[str] = []
        for row in rows[:_GROUND_RESERVE]:
            ids.append(row["chunk_id"])
            if row["chunk_id"] not in tags:
                reserved.append((ground, row))
            tags[row["chunk_id"]].append((ground, row))
        reserved_ids.append(ids)

    def appended_for(m: int) -> list[tuple[str, dict[str, Any]]]:
        kept = {row["chunk_id"] for row in main[:m]}
        return [(ground, row) for ground, row in reserved if row["chunk_id"] not in kept]

    m = min(len(main), pool_size)
    # m + |appended(m)| never decreases as m grows, so the first fit walking down is the largest.
    while m > 0 and m + len(appended_for(m)) > pool_size:
        m -= 1
    appended = appended_for(m)[:pool_size - m]  # truncates only when the reserve alone exceeds the pool

    pool = [dict(row) for row in main[:m]] + [dict(row) for _ground, row in appended]
    for row in pool:
        for ground, ground_row in tags.get(row["chunk_id"], ()):
            _union_ground(row, ground, ground_row)
    for rank, row in enumerate(pool, start=1):
        row["fusion_rank"] = rank

    appended_by_ground: dict[str, list[str]] = defaultdict(list)
    for ground, row in appended:
        appended_by_ground[ground].append(row["chunk_id"])
    details = [
        {"ground": ground, "reserved_chunk_ids": ids, "appended_chunk_ids": appended_by_ground[ground]}
        for (ground, _rows), ids in zip(grounds, reserved_ids)
    ]
    return pool, m, details


def _count_by_method(candidates: list[dict[str, Any]], method: str) -> int:
    return sum(1 for candidate in candidates if method in (candidate.get("retrieval_methods") or []))


def _pool_provenance(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries = []
    for candidate in candidates:
        entry = {
            "chunk_id": candidate["chunk_id"],
            "fusion_rank": candidate.get("fusion_rank"),
            "retrieval_methods": list(candidate.get("retrieval_methods") or []),
            "perspectives": list(candidate.get("perspectives") or []),
        }
        if candidate.get("retrieval_grounds"):
            entry["retrieval_grounds"] = list(candidate["retrieval_grounds"])
        entries.append(entry)
    return entries


@dataclass
class CandidatePool:
    """The reranker's input pool, its provenance, and the first vector-search error type (if any)."""

    candidates: list[dict[str, Any]]
    details: dict[str, Any]
    vector_error_type: str | None = None


def candidate_pool(
    index: AgreementIndex, documents: list[Any], *, auditor: str, shell: str = "",
    vectors: Mapping[str, Sequence[float]] | None = None, limit: int | None = None,
) -> CandidatePool:
    """Steps 1, 3 and 4: run every search, rank each, merge the main searches, add the ground reserve.

    ``vectors`` maps a search text to its query vector (``search_texts`` lists the texts); a search
    without one is lexical only. No LLM or network call happens here; the only I/O is through the
    read-only ``AgreementIndex``.
    """
    searches, grounds = _all_searches(auditor, shell)
    vectors = vectors or {}
    pool_size = limit if limit is not None else _max_candidates()
    start = time.perf_counter()
    results = [
        _retrieve_perspective(index, search, documents, vectors.get(search.text)) for search in searches
    ]
    ground_results = [
        _retrieve_perspective(index, search, documents, vectors.get(search.text)) for search in grounds
    ]
    # Ground searches never enter the cross-search RRF; they only feed the reserve.
    main, cross_search_ranking = _merge_candidates([result.rows for result in results], None)
    candidates, main_pool_count, ground_details = _assemble_pool(
        main,
        [(search.source.removeprefix("ground:"), result.rows)
         for search, result in zip(grounds, ground_results)],
        pool_size,
    )
    for search, entry in zip(grounds, ground_details):
        entry["perspective"] = search.perspective
        entry["perspectives"] = search.row_perspectives()
    details = {
        "search_queries": [
            {"text": search.text, "source": search.source, "perspective": search.perspective}
            for search in (*searches, *grounds)
        ],
        "per_search": [result.provenance() for result in (*results, *ground_results)],
        "cross_search_ranking": cross_search_ranking,
        "ground_searches": [
            {key: entry[key] for key in ("ground", "perspective", "perspectives", "reserved_chunk_ids",
                                         "appended_chunk_ids")}
            for entry in ground_details
        ],
        "ground_reserve": _GROUND_RESERVE,
        "pool_size": pool_size,
        "main_pool_count": main_pool_count,
        "candidate_pool_count": len(candidates),
        "candidate_count_lexical": _count_by_method(candidates, "lexical"),
        "candidate_count_vector": _count_by_method(candidates, "vector"),
        "candidate_pool": _pool_provenance(candidates),
        "candidate_generation_seconds": round(time.perf_counter() - start, 3),
    }
    error_type = next((result.vector_error_type for result in (*results, *ground_results)
                       if result.vector_error_type), None)
    return CandidatePool(candidates=candidates, details=details, vector_error_type=error_type)


# --- Rerank and selection -------------------------------------------------------------------------

def _model_info(node: str) -> dict[str, str]:
    tier = llm.tier_for(node)
    try:
        model = llm._xllm.model_for_tier(tier)
    except Exception:  # provider not configured (offline tests)
        model = ""
    return {"node": node, "tier": tier, "model": model}


def _candidate_row(candidate: dict[str, Any]) -> dict[str, Any]:
    """Prompt view of one candidate: source metadata and an excerpt, never retrieval scores."""
    return {
        "chunk_id": candidate["chunk_id"],
        "document_name": candidate.get("document_name") or "",
        "document_type": candidate.get("document_type") or "",
        "legal_track": candidate.get("legal_track") or "",
        "heading_path": candidate.get("heading_path") or "",
        "pdf_pages": [candidate.get("pdf_page_start"), candidate.get("pdf_page_end")],
        "party_scope": candidate.get("party_scope") or [],
        "retrieved_for": candidate.get("perspectives") or [],
        "excerpt": _candidate_excerpt(candidate.get("verbatim_text") or ""),
    }


def _ranking_prompt(
    *, asset: str, year: str, tracks: list[str], auditor: str, shell: str,
    candidates: list[dict[str, Any]],
) -> str:
    rows = [_candidate_row(candidate) for candidate in candidates]
    return (
        f"CANONICAL ASSET: {asset}\n"
        f"CASE YEAR: {year or '(unknown)'}\n"
        f"APPLICABLE LEGAL TRACKS: {json.dumps(tracks)}\n\n"
        f"AUDITOR THEORY (original exception text):\n{auditor or '(unavailable)'}\n\n"
        f"ADDITIVE SHELL FINDINGS:\n{shell or '(no specialist findings available)'}\n\n"
        f"CANDIDATES:\n{json.dumps(rows, ensure_ascii=False)}\n\n"
        "Label every candidate that bears on the dispute. Return IDs and labels only; do not reproduce "
        "source text."
    )


def _validated_labels(
    ranking: Any, candidates: dict[str, dict[str, Any]],
) -> list[tuple[_RankedCandidate, dict[str, Any]]]:
    """Reranker output restricted to known IDs and labels, in reranker order (not_relevant kept)."""
    parsed = _AgreementRanking.model_validate(_dump(ranking))
    labeled: list[tuple[_RankedCandidate, dict[str, Any]]] = []
    seen: set[str] = set()
    for item in parsed.ranked_candidates:
        if item.chunk_id in seen or item.chunk_id not in candidates:
            continue
        relevance = item.relevance.strip().lower()
        relevance = _LABEL_ALIASES.get(relevance, relevance)
        if relevance not in _LABEL_ORDER and relevance != "not_relevant":
            continue
        item.relevance = relevance
        relationship = item.relationship.strip().lower()
        item.relationship = relationship if relationship in _RELATIONSHIPS else "context"
        item.perspectives = [
            value for value in dict.fromkeys(item.perspectives) if value in _PERSPECTIVES
        ] or list(candidates[item.chunk_id].get("perspectives") or [])
        labeled.append((item, candidates[item.chunk_id]))
        seen.add(item.chunk_id)
    return labeled


def _near_duplicate(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left["document_id"] != right["document_id"]:
        return False
    left_words = set(re.findall(r"[a-z0-9]+", left["verbatim_text"].casefold()))
    right_words = set(re.findall(r"[a-z0-9]+", right["verbatim_text"].casefold()))
    if not left_words or not right_words:
        return False
    return len(left_words & right_words) / len(left_words | right_words) >= 0.85


def _select_ranked(
    labeled: list[tuple[_RankedCandidate, dict[str, Any]]], limit: int,
) -> list[tuple[_RankedCandidate, dict[str, Any]]]:
    """Top `limit` by label (governing > supporting > auditor_cited), then reranker order.

    No per-track or per-perspective quota; a near-duplicate of an already selected chunk is skipped.
    """
    eligible = [entry for entry in labeled if entry[0].relevance in _LABEL_ORDER]
    eligible.sort(key=lambda entry: _LABEL_ORDER[entry[0].relevance])  # stable: keeps reranker order
    selected: list[tuple[_RankedCandidate, dict[str, Any]]] = []
    for entry in eligible:
        if len(selected) >= limit:
            break
        if any(_near_duplicate(entry[1], current[1]) for current in selected):
            continue
        selected.append(entry)
    return selected


def rank_pool(
    *, asset: str, year: str, tracks: list[str], auditor: str, shell: str,
    candidates: list[dict[str, Any]], context: str = "", limit: int | None = None,
) -> tuple[list[tuple[_RankedCandidate, dict[str, Any]]], dict[str, Any]]:
    """Steps 5-6: one listwise rerank over the whole pool, then label-ordered selection.

    `candidates` are decoded chunk rows in pool order. Returns the selected (label, candidate) pairs
    and provenance details. Source text is NOT hydrated here; callers re-read IDs from SQLite.
    Shared by the node and `otl.rag_tests.replay_rerank`, so replays run the exact production logic.
    """
    prompt = _ranking_prompt(
        asset=asset, year=year, tracks=tracks, auditor=auditor, shell=shell, candidates=candidates,
    )
    start = time.perf_counter()
    ranking = llm.structured("agreement_retriever", _RERANK_SYSTEM, prompt + context, _AgreementRanking)
    details: dict[str, Any] = {"rerank_seconds": round(time.perf_counter() - start, 3)}
    labeled = _validated_labels(ranking, {candidate["chunk_id"]: candidate for candidate in candidates})
    details["rerank_model"] = _model_info("agreement_retriever")
    details["rerank_labels"] = [
        {"chunk_id": item.chunk_id, "relevance": item.relevance} for item, _candidate in labeled
    ]
    return _select_ranked(labeled, limit if limit is not None else top_k()), details


# --- Node -----------------------------------------------------------------------------------------

def _empty_result(
    state: dict, *, status: str, reason: str, canonical: str = "", auditor: str = "",
    shell: str = "", documents: list[Any] | None = None, provenance: dict | None = None,
) -> dict:
    docs = documents or []
    prov = provenance or {}
    sub = AgreementRetrievalState(
        nl_summary=reason,
        status=status,
        canonical_asset=canonical,
        auditor_query=auditor,
        shell_findings_query=shell,
        applicable_documents=[doc.document_id for doc in docs],
        applicable_tracks=list(dict.fromkeys(doc.legal_track for doc in docs)),
        index_provenance=prov,
    )
    update = {"agreement_retriever": sub}
    update.update(trace_event(
        state, "agreement_retriever", "completed agreement retrieval",
        inputs_digest=(
            "auditor exception + additive specialist findings; asset-filtered agreement index; "
            f"effective_mode={prov.get('effective_mode', 'bm25_only')}"
        ),
        key_outputs={"status": status, "canonical_asset": canonical, "selected_chunks": 0,
                     "reason": reason},
        nl_summary=reason,
    ))
    return update


def _trace_digest(canonical: str, provenance: dict[str, Any]) -> str:
    searches = ", ".join(
        f"{search.get('source') or search['perspective']}={search['ranking']}"
        for search in provenance.get("per_search") or []
    )
    return (f"asset={canonical}; searches=[{searches}]; "
            f"cross_search={provenance.get('cross_search_ranking')}; "
            f"pool={provenance.get('candidate_pool_count')} "
            f"(main={provenance.get('main_pool_count')}); "
            f"reranked={len(provenance.get('rerank_labels') or [])}; "
            f"effective_mode={provenance.get('effective_mode')}")


def agreement_retriever_node(state: dict) -> dict:
    auditor = auditor_query(state)
    shell = shell_findings_query(state)
    case = state.get("case") or {}
    case_asset = str(case.get("asset") or "")
    case_year = str(case.get("year") or "")
    canonical = ""
    documents: list[Any] = []
    requested_mode, effective_mode, fallback_reason, fallback_error_type = _resolve_mode()
    provenance: dict[str, Any] = _mode_fields(
        requested_mode, effective_mode, fallback_reason, fallback_error_type,
    )
    limit = top_k()
    provenance["top_k"] = limit
    provenance["date_filter"] = date_filter_provenance(case_year, [])
    try:
        manifest = load_manifest()
        canonical = canonical_asset(case_asset, manifest)
        if not canonical:
            return _empty_result(
                state, status="unsupported_asset",
                reason=f"no agreement corpus mapping for asset {case_asset!r}", auditor=auditor,
                shell=shell, provenance=provenance,
            )
        # Effective-date filter: only documents active in the audit year are searched or reported.
        documents, excluded = documents_for_case(canonical, case_year, manifest)
        provenance["date_filter"] = date_filter_provenance(case_year, excluded)
        if excluded and not documents:
            return _empty_result(
                state, status="no_match",
                reason=(f"no agreement for asset {canonical!r} is effective in audit year "
                        f"{provenance['date_filter']['year']} (all {len(excluded)} excluded by "
                        "active_start_date/active_end_date)"),
                canonical=canonical, auditor=auditor, shell=shell, documents=documents,
                provenance=provenance,
            )
        tracks = list(dict.fromkeys(doc.legal_track for doc in documents))
        index = AgreementIndex()
        provenance.update(index.metadata())
        provenance["index_path"] = str(index.path)
        if not index.exists():
            return _empty_result(
                state, status="unavailable", reason="agreement index is not available",
                canonical=canonical, auditor=auditor, shell=shell, documents=documents,
                provenance=provenance,
            )
        if not index.ready_for(manifest):
            return _empty_result(
                state, status="stale", reason="agreement index does not match the current manifest",
                canonical=canonical, auditor=auditor, shell=shell, documents=documents,
                provenance=provenance,
            )
        if not auditor and not shell:
            return _empty_result(
                state, status="no_match", reason="no auditor or specialist retrieval query was available",
                canonical=canonical, auditor=auditor, shell=shell, documents=documents,
                provenance=provenance,
            )

        # Step 2 (hybrid only): compatibility (no Azure call) -> one embedding call for every search.
        # Any failure leaves every search without a vector; the pipeline below runs regardless.
        vectors: dict[str, list[float]] = {}
        if requested_mode == "hybrid":
            try:
                settings = _check_semantic_compatibility(index)
                semantic = _embed_queries(settings, search_texts(auditor, shell))
                vectors = semantic.vectors_by_text
                provenance.update(semantic.provenance())
            except _Fallback as fallback:
                fallback_reason, fallback_error_type = fallback.reason, fallback.error_type

        # Steps 3-4: per-search ranking, cross-search merge, pool cut.
        pool = candidate_pool(index, documents, auditor=auditor, shell=shell, vectors=vectors)
        if pool.vector_error_type and fallback_reason is None:
            fallback_reason, fallback_error_type = "vector_search_failed", pool.vector_error_type
        effective_mode = _effective_mode(requested_mode, pool.details["per_search"])
        provenance.update(_mode_fields(requested_mode, effective_mode, fallback_reason, fallback_error_type))
        provenance.update(pool.details)
        if not pool.candidates:
            return _empty_result(
                state, status="no_match", reason="no relevant agreement candidates were retrieved",
                canonical=canonical, auditor=auditor, shell=shell, documents=documents,
                provenance=provenance,
            )

        # Steps 5-6: one listwise rerank, label-ordered selection.
        selected, details = rank_pool(
            asset=canonical, year=case_year, tracks=tracks, auditor=auditor, shell=shell,
            candidates=pool.candidates, context=rebuttal_context_block(state), limit=limit,
        )
        provenance.update(details)
        by_id = {candidate["chunk_id"]: candidate for candidate in pool.candidates}
        qualifying_tracks = list(dict.fromkeys(
            by_id[row["chunk_id"]]["legal_track"] for row in details["rerank_labels"]
            if row["relevance"] in _LABEL_ORDER
        ))
        # Step 7: hydrate the selected IDs from SQLite.
        return _selection_result(
            state, index=index, selected=selected, qualifying_tracks=qualifying_tracks,
            canonical=canonical, documents=documents, tracks=tracks, auditor=auditor, shell=shell,
            provenance=provenance, inputs_digest=_trace_digest(canonical, provenance),
        )
    except Exception as exc:
        return _empty_result(
            state, status="error", reason=f"agreement retrieval failed: {type(exc).__name__}: {exc}",
            canonical=canonical, auditor=auditor, shell=shell, documents=documents,
            provenance=provenance,
        )


def _selection_result(
    state: dict, *, index: AgreementIndex, selected: list[tuple[Any, dict[str, Any]]],
    qualifying_tracks: list[str], canonical: str, documents: list[Any], tracks: list[str],
    auditor: str, shell: str, provenance: dict[str, Any], inputs_digest: str,
) -> dict:
    """Re-read the selected IDs from SQLite and build the node update."""
    if not selected:
        return _empty_result(
            state, status="no_match", reason="the reranker found no materially relevant agreement chunks",
            canonical=canonical, auditor=auditor, shell=shell, documents=documents,
            provenance=provenance,
        )

    # Source-text boundary: re-read the selected IDs from SQLite.  Neither the LLM output nor the
    # candidate prompt is trusted as the final quote text.
    hydrated = index.get_chunks(item.chunk_id for item, _candidate in selected)
    chunks: list[AgreementChunk] = []
    for item, _candidate in selected:
        source = hydrated.get(item.chunk_id)
        if not source:
            continue
        chunks.append(AgreementChunk(
            **source,
            relevance=item.relevance,
            perspectives=item.perspectives,
            relationship=item.relationship,
            selection_reason=item.selection_reason,
        ))
    covered = list(dict.fromkeys(chunk.legal_track for chunk in chunks))
    uncovered = [track for track in qualifying_tracks if track not in covered]
    status = "matched" if chunks else "error"
    summary = (
        f"selected {len(chunks)} agreement chunk(s) across {len(covered)} legal track(s)"
        if chunks else "selected agreement chunks could not be rehydrated from the index"
    )
    matching = _dump(state.get("matching"))
    authority_text = "\n".join(chunk.verbatim_text for chunk in chunks).casefold()
    non_chargeable = any(pattern in authority_text for pattern in (
        "not chargeable", "non-chargeable", "non chargeable", "shall not be charged",
        "not be charged", "covered by overhead", "not billable", "non-billable",
    ))
    chargeable = any(pattern in authority_text for pattern in (
        "direct charged", "charged direct", "chargeable", "shall be charged", "billable",
    ))
    if non_chargeable and not chargeable:
        copas_resolution = "supports_non_chargeability"
    elif chargeable and not non_chargeable:
        copas_resolution = "supports_chargeability"
    else:
        copas_resolution = "unclear"
    threshold_hours: float | None = None
    threshold_match = re.search(
        r"(?:less than|minimum(?: of)?|at least)?\s*(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)"
        r"[^.]{0,100}(?:per|in each|each)\s+(?:calendar\s+)?month",
        authority_text,
    )
    if threshold_match:
        threshold_hours = float(threshold_match.group(1))
    authority_rationale = (
        "Deterministic classification of the exact rehydrated selected authority; "
        f"chargeable_language={chargeable}, non_chargeable_language={non_chargeable}."
    )
    if copas_resolution == "supports_non_chargeability":
        agreement_chargeability = "not_chargeable"
    elif copas_resolution == "supports_chargeability":
        agreement_chargeability = "chargeable"
    else:
        agreement_chargeability = "unclear"
    citation_refs = [{
        "chunk_id": chunk.chunk_id,
        "document_name": chunk.document_name,
        "legal_track": chunk.legal_track,
        "heading_path": chunk.heading_path,
        "pdf_page_start": chunk.pdf_page_start,
        "pdf_page_end": chunk.pdf_page_end,
        "printed_page_labels": list(chunk.printed_page_labels),
        "quote_safe": chunk.quote_safe,
    } for chunk in chunks]

    sub = AgreementRetrievalState(
        nl_summary=summary,
        status=status,
        canonical_asset=canonical,
        auditor_query=auditor,
        shell_findings_query=shell,
        applicable_documents=[doc.document_id for doc in documents],
        applicable_tracks=tracks,
        covered_tracks=covered,
        uncovered_tracks=uncovered,
        selected_chunks=chunks,
        index_provenance=provenance,
        copas_mfi_resolution=copas_resolution,
        cfr_copas_resolution="unclear",
        agreement_chargeability=agreement_chargeability,
        agreement_chargeability_citations=citation_refs,
        authority_rationale=authority_rationale,
        mcoh_threshold_hours=threshold_hours,
        mcoh_threshold_source="agreement" if threshold_hours is not None else "configured_default",
    )
    update = {"agreement_retriever": sub}
    update.update(evaluate_state({**state, **update}, final=False))
    update.update(trace_event(
        state, "agreement_retriever", "selected top agreement chunks",
        inputs_digest=inputs_digest,
        key_outputs={
            "status": status,
            "selected_chunk_ids": [chunk.chunk_id for chunk in chunks],
            "covered_tracks": covered,
            "top_k": provenance["top_k"],
            "source_text_hydrated_after_ranking": True,
            "copas_mfi_resolution": copas_resolution,
            "agreement_chargeability": agreement_chargeability,
            "agreement_chargeability_citations": citation_refs,
            "mcoh_threshold_hours": threshold_hours,
        },
        nl_summary=summary,
    ))
    return update


__all__ = [
    "MAX_CANDIDATES", "TOP_K", "CandidatePool", "agreement_retriever_node", "auditor_query",
    "candidate_pool", "rank_pool", "search_texts", "shell_findings_query", "top_k",
]
