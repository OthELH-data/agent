"""Deterministic COPAS error-code classifier ("poor man's classifier").

Every audit exception carries an auditor-assigned COPAS error code (field
`error_type`, e.g. "03.1 Material/Equipment Reconciliations"). Because that
vocabulary is exhaustive and controlled, category is a LOOKUP, not an inference.

This module parses the numeric code prefix (robust to truncated labels) and maps
it to {code, label, category, family, tier_hint} via registry/copas_error_codes.yaml.
"""
from __future__ import annotations
import re, functools, yaml
from pathlib import Path

_REGISTRY = Path(__file__).resolve().parent / "copas_error_codes.yaml"

# Matches "03.1", "R21.2", "1.0", "06.3 Admin..." -> captures the code token.
_CODE_RE = re.compile(r"^\s*(R?\d{1,2}\.\d)", re.I)


@functools.lru_cache(maxsize=1)
def _codes() -> dict:
    d = yaml.safe_load(_REGISTRY.read_text())
    return d.get("codes", {})


def parse_code(error_type: str) -> str:
    """Extract the canonical code token ('03.1', 'R21.2') from a raw error_type
    string. Returns '' if no code can be parsed."""
    if not error_type:
        return ""
    m = _CODE_RE.match(str(error_type).strip())
    if not m:
        return ""
    tok = m.group(1).upper()
    # normalize "1.0" -> "01.0" (registry keys are zero-padded)
    if re.match(r"^\d\.\d$", tok):
        tok = "0" + tok
    return tok


def classify(error_type: str) -> dict:
    """Return {code, label, category, family, tier_hint, matched} for an error_type.
    Deterministic; never calls an LLM. Unknown/blank -> Unclassified (matched=False)."""
    code = parse_code(error_type)
    reg = _codes()
    if code and code in reg:
        e = reg[code]
        return {"code": code, "label": e.get("label", ""), "category": e.get("category", "Other"),
                "family": e.get("family", "misc"), "tier_hint": e.get("tier_hint", 2), "matched": True}
    return {"code": code, "label": "", "category": "Unclassified", "family": "unknown",
            "tier_hint": 2, "matched": False}


if __name__ == "__main__":
    import json, collections
    data = json.load(open(Path(__file__).resolve().parent.parent / "data/exceptions/exceptions_classified.json"))
    n = len(data)
    matched = 0
    cats = collections.Counter()
    unmatched = collections.Counter()
    for e in data:
        r = classify(e.get("error_type", ""))
        cats[r["category"]] += 1
        if r["matched"]:
            matched += 1
        else:
            unmatched[e.get("error_type", "(blank)")] += 1
    print(f"Records: {n}  |  matched to a code: {matched} ({100*matched/n:.1f}%)")
    print("\n=== category distribution (deterministic) ===")
    for c, v in cats.most_common():
        print(f"  {v:4d}  {c}")
    if unmatched:
        print("\n=== UNMATCHED error_type values (need registry entry) ===")
        for u, v in unmatched.most_common():
            print(f"  {v:4d}  {u!r}")
