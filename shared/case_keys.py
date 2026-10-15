import csv
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

def _norm(s: str) -> str:
    """Collapse an asset name for folder matching (CSV shorthand 'GW' -> 'greatwhite')."""
    s = (s or "").lower()
    s = re.sub(r"\bgw\b", "greatwhite", s)
    return re.sub(r"[^a-z0-9]", "", s)


def pad_exc(exc_id: str) -> str:
    """Canonicalize an exception id to the corpus's 4-digit form ('40' -> '0040').

    Every exception directory in the output tree is 4-digit, and `exc_id` lands there verbatim
    (`<year>/<asset>/<exc_id>/`) — so padding here is what stops the same exception resolving to two
    different case ids (and two different output paths) depending on which CSV named it. Non-numeric ids
    ('Ex 3') are left untouched.
    """
    s = (exc_id or "").strip()
    # Excel exports often serialize integer IDs as e.g. ``3.0``.
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".", 1)[0]
    return s.zfill(4) if s.isdigit() and len(s) < 4 else s
    

__all__ = ["_norm", "pad_exc"]