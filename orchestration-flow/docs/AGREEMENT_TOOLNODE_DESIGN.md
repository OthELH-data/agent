# Deferred Agreement ToolNode Design

Status: **Deferred. Not part of the current implementation.**

The intended later architecture replaces aggregate-only agreement retrieval with a shared native
LangGraph ToolNode that specialists may call for narrowly scoped legal questions. Each tool-using
specialist becomes a subgraph with an analyze → tool → resume loop, while the main orchestrator still
sees one specialist visit.

The deferred design includes:

- Typed agreement-search requests and responses.
- Trusted injection of asset, year, parties, documents, and legal tracks.
- A provider-neutral tool-capable LLM adapter.
- Neutral-query and grounding validation through an AgreementQueryPolicyHook.
- Completion validation through an AuthorityCoverageHook.
- Exact source-chunk hydration after candidate-ID validation.
- Per-question caching and batching rather than one search per person.
- An append-only agreement retrieval registry shared across agents.
- A final authority-gap completion subgraph before precedent review and drafting.
- Shadow-mode comparison and staged migration away from the aggregate terminal query.

The current implementation keeps the existing terminal agreement retriever and its aggregate auditor
and Shell-findings queries.
