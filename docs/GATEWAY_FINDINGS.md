# Gateway usage-accounting findings (Probes A–F)

**Measured:** 2026-08-10 · Anthropic Python SDK `0.75.0` via `base_url` (no OpenAI shim, no tiktoken).

## Which gateway these numbers come from — read this first

Two LiteLLM deployments are in play:

| | Project gateway (target) | Probe gateway (measured) |
|---|---|---|
| Base URL | `https://api.nextgen-beta.ica.ibm.com/ica/v1` | sandbox `ANTHROPIC_BASE_URL` |
| Credential | `ICA_API_KEY` — **REVOKED**, HTTP 401 `token_not_found_in_db` on every route | valid |
| Software | LiteLLM (401 body names `LiteLLM_VerificationTokenTable`) | LiteLLM |
| Model IDs | `claude-opus-4-8` / `claude-sonnet-4-6` / `claude-haiku-4-5` | **same three IDs** |

Probes A–F ran against the **probe gateway**. Same proxy software and same model IDs, but
cache-token semantics are a per-deployment LiteLLM config detail. **Re-run
`scripts/probe_gateway_usage.py` against ICA once the key is restored to confirm.**
That is exactly why `effective_input_tokens` detects inclusive/exclusive **at runtime**
instead of hardcoding what was seen here — if ICA differs, the code is still correct.

## Correction to this repo's own docs

`shared/tokens.py` states *"The ICA gateway is OpenAI-spec compatible."*
That is **wrong**, or at least incomplete. Route probing on the ICA host (unauthenticated,
distinguishing 401 "no api key" from 404 "Not Found") shows the Anthropic-native surface is
mounted: `POST /messages` → 401, `POST /messages/count_tokens` → 401, `POST /chat/completions`
→ 401, while `/openapi.json`, `/docs`, `/` → 404. The 401s are real route hits, so the
Anthropic SDK is a valid client against ICA.

## Probe results

| Probe | Question | Result |
|---|---|---|
| **A** | Field naming, null vs 0 | **Anthropic names only.** No `prompt_tokens` / `completion_tokens` / `prompt_tokens_details` anywhere. Wire: `input_tokens=14`, `output_tokens=4`, `cache_creation_input_tokens=0`, `cache_read_input_tokens=0`, `cache_creation={ephemeral_5m_input_tokens:0, ephemeral_1h_input_tokens:0}`. `null` fields (`inference_geo`, `server_tool_use`, `service_tier`) are **SDK-synthesised, not on the wire** — proven by comparing raw HTTP JSON against `model_dump()`. So `0` = reported-as-zero, `null` = never sent. |
| **B** | Does usage survive streaming, and where | **Yes, no `include_usage` needed.** `message_start` already carries the **complete input side** (`input_tokens=14` + both cache counters) with `output_tokens=1` provisional; `message_delta` carries final `output_tokens=4`; `final_message` (via `stream.get_final_message()`) has both. **Input-side accounting is safe to read at `message_start`; output is not.** |
| **C** | Cache token semantics | **EXCLUSIVE — caching works.** Write call: `input_tokens=12`, `cache_creation=12943`, `cache_read=0`, nested `ephemeral_5m_input_tokens=12943`. Hit call: `input_tokens=12`, `cache_read=12943`, `cache_creation=0`. Since `12 < 12943 + 0` → exclusive. **Full prompt = `input + read + creation` = 12,955.** Using `input_tokens` alone would report **12 tokens for a 12,955-token prompt** — a 1,079× undercount, and a %-of-context threshold would never fire. |
| **D** | Is `count_tokens` proxied? | **Yes, and semantically real, but it does NOT agree with billing.** Constant **+17** offset vs `messages.create` across four payloads (31/14, 25/8, 25/8, 25/8 — delta `+17` every time). It honours `system` (14,027 for a large system prompt) and `tools` (580 vs 25). Verdict: **usable as a conservative pre-flight counter** (it over-counts, which is the safe direction), **not** authoritative for reconciling billed usage. |
| **E** | Overflow error shape | **Could not reproduce overflow — and that is itself the finding.** Accepted **576,198 input tokens** with no error. Needle test rules out silent truncation: a passphrase at the very front of a **523,860-token** prompt was recovered verbatim, so context is genuinely served, not clipped. Real limit is 1M (Probe F), so a true overflow costs ~1M tokens/attempt — deliberately not run. **Real error shapes captured instead** (below). |
| **F** | Context-window ground truth | `GET /v1/models/{id}` **is** proxied but returns `max_input_tokens: null`, `max_tokens: null` — useless. **`GET /v1/model/info` works**: `max_input_tokens=1000000`, `max_output_tokens=64000`, `max_tokens=8192`. |

## Probe E detail — what the gateway's errors actually look like

Captured live (`max_tokens` too large, HTTP **400**):

```
{"message":"The maximum tokens you requested exceeds the model limit of 128000"}. Received Model Group=claude-sonnet-4-6
Available Model Group Fallbacks=None
```

Unknown model, HTTP **403**, `error.type = "team_model_access_denied"`.
Negative `max_tokens`, HTTP **400**: `{"message":"max_tokens: must be greater than or equal to 0"}. Received Model Group=...`

Three structural lessons for the reactive matcher:

1. **LiteLLM nests the upstream error as a JSON *string* inside `error.message`**, then appends
   its own `Received Model Group=… / Available Model Group Fallbacks=…` trailer. The real text is
   double-encoded.
2. **`error.type` is `None`** on these 400s. Matching `type == "invalid_request_error"` — the
   first-party contract — **fails on this gateway**.
3. Matching first-party wording `"prompt is too long"` would also fail. Match **case-insensitive
   substrings against the whole flattened error text**, and treat HTTP 400 + a size-ish phrase as
   overflow. That is what `is_context_overflow_error()` does; its substring list includes
   both first-party and LiteLLM phrasings.

Because a genuine input overflow was never observed, the matcher is **necessarily
speculative on the exact input-overflow string**. It is deliberately broad, and the
80%-of-limit proactive threshold — not the matcher — is the primary defence.

## Limits actually used

| Value | Number | Source |
|---|---|---|
| `claude-sonnet-4-6` input limit | **1,000,000** | measured, `GET /v1/model/info` |
| Compaction trip point (80%) | 800,000 | derived |
| Warn tier (60%) | 600,000 | derived |

⚠️ **Sources disagree on output limits**: `/v1/model/info` says `max_output_tokens=64000` and
`max_tokens=8192`, while the live 400 error says *"model limit of 128000"*. Input-side is what
matters here and is consistent, but do not trust a single source for output caps.

## Consequences for the code

- **Never sum all three cache fields unconditionally.** Correct under exclusive, ~2× over under
  inclusive. Decide per-response with `input_tokens >= read + creation`.
- **A 12-token `input_tokens` is not a small prompt** — under exclusive semantics it means a
  large cached prefix. This is the failure mode that silently disables compaction.
- **`cache_creation` nested sub-object** (`ephemeral_5m` / `ephemeral_1h`) is not in the flat
  field list. `capture_usage` serialises the whole usage object generically so TTL-bucket splits
  survive.
- **tiktoken `cl100k_base`, used by `tokens.py` for every `claude-*` id, is the wrong tokenizer**
  and undercounts Claude ~15–20% on prose, more on code. Replaced by gateway-reported usage,
  with a documented character heuristic (not an OpenAI encoding) as the fallback.
- **Prompt-cache minimum:** documented 1,024 for Sonnet 4.6, but
  [anthropic-sdk-python#1194](https://github.com/anthropics/anthropic-sdk-python/issues/1194)
  reports writes not landing until ~2,048. Probe C used a ~12,964-token system prompt (4× the
  larger figure) so it cannot fail for being too small.
