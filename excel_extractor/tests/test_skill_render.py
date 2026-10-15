"""Tests for `render_analysis` — the synthesis-prompt findings block.

The bug these guard against: findings are produced in exhibit row order, and the renderer used to cut
the list to a character budget from the front. On GreatWhite_2023_0003 that kept 17 of 32 findings, so
the synthesis reported the charge window as periods 003.2022-008.2022 (March-August) when the exhibit
actually ran 003.2022-012.2022, and reported $27,499.67 against a true column total of $47,365.25. The
loss was invisible downstream: an early-stopping date range and a short total read as a complete answer.

So the contract under test is a partition, not a fidelity claim: every finding must end up either in
the verbatim tail or in the batch handed to the summarizer. Nothing may be silently discarded.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extractor import skill_engine  # noqa: E402


def _analysis(findings, **extra):
    """A minimal apply_skill-shaped result."""
    base = {
        "skill_name": "test-skill",
        "findings": findings,
        "findings_summary": "",
        "summarized_upto": 0,
        "sufficient": True,
        "chunks_scanned": 1,
        "total_chunks": 1,
        "required_tabs": [],
        "covered_tabs": [],
        "gaps": [],
    }
    base.update(extra)
    return base


class _StubTextCall:
    """Stands in for the LLM fold. Records the batch it was given and echoes it back."""

    def __init__(self, reply=None):
        self.calls = []
        self.reply = reply

    def __call__(self, tier, system, user):
        self.calls.append({"tier": tier, "system": system, "user": user})
        return self.reply if self.reply is not None else f"FOLDED::{user}"

    @property
    def folded_text(self):
        """Everything ever handed to the summarizer, concatenated."""
        return "\n".join(c["user"] for c in self.calls)


class RenderAnalysisTest(unittest.TestCase):
    def setUp(self):
        self._real_text_call = skill_engine.text_call
        self.stub = _StubTextCall()
        skill_engine.text_call = self.stub

    def tearDown(self):
        skill_engine.text_call = self._real_text_call

    # -- the fitting case: no compression, no LLM call ---------------------------------------------

    def test_all_findings_render_verbatim_when_within_budget(self):
        findings = [f"Sheet1 row {i}: period 0{i:02d}.2022 amount {i * 100}.00" for i in range(1, 13)]
        out = skill_engine.render_analysis(_analysis(findings), max_chars=100_000)
        for f in findings:
            self.assertIn(f, out)
        self.assertEqual(self.stub.calls, [], "no LLM fold should happen when everything fits")
        self.assertIn("all 12 findings", out)

    def test_no_elision_marker_is_ever_emitted(self):
        findings = [f"finding {i} " + "x" * 400 for i in range(40)]
        out = skill_engine.render_analysis(_analysis(findings), max_chars=2_000)
        self.assertNotIn("elided", out)
        self.assertNotIn("omitted", out)

    # -- the overflow case: compression, and a complete partition ----------------------------------

    def test_every_finding_survives_when_over_budget(self):
        findings = [f"[M{i:03d}] Sheet1 row {i}: amount {i * 37}.25" for i in range(60)]
        out = skill_engine.render_analysis(_analysis(findings), max_chars=1_500)
        combined = out + "\n" + self.stub.folded_text
        missing = [f for f in findings if f not in combined]
        self.assertEqual(missing, [], f"{len(missing)} findings reached neither the tail nor the fold")

    def test_newest_findings_are_the_ones_kept_verbatim(self):
        """The old renderer kept the oldest. Later rows carry the later dates, so keep those."""
        self.stub.reply = "<<SUMMARY>>"      # non-echoing, so the verbatim tail is unambiguous
        findings = [f"[M{i:03d}] " + "y" * 300 for i in range(30)]
        out = skill_engine.render_analysis(_analysis(findings), max_chars=1_200)
        self.assertIn("[M029]", out, "the last finding must always be present verbatim")
        self.assertNotIn("[M000]", out, "the oldest should have been folded, not rendered")
        self.assertIn("[M000]", self.stub.folded_text, "...but it must still reach the summarizer")

    def test_true_finding_count_is_always_stated(self):
        findings = [f"finding {i} " + "z" * 300 for i in range(47)]
        out = skill_engine.render_analysis(_analysis(findings), max_chars=1_000)
        self.assertIn("47 findings", out)

    def test_fold_receives_a_contiguous_head(self):
        findings = [f"[M{i:03d}] " + "w" * 300 for i in range(25)]
        skill_engine.render_analysis(_analysis(findings), max_chars=1_200)
        self.assertEqual(len(self.stub.calls), 1)
        sent = self.stub.calls[0]["user"]
        seen = [i for i in range(25) if f"[M{i:03d}]" in sent]
        self.assertEqual(seen, list(range(len(seen))), "folded batch must be a prefix, in order")

    # -- reuse of the scan's own rolling fold ------------------------------------------------------

    def test_scan_fold_is_reused_instead_of_recomputed(self):
        # 311 chars per rendered finding, so a 1,200-char budget holds 3 verbatim → split is 27.
        # summarized_upto=28 is past that, so the scan's fold already covers what the budget needs.
        findings = [f"[M{i:03d}] " + "q" * 300 for i in range(30)]
        analysis = _analysis(findings, findings_summary="PRIOR ROLLING SUMMARY", summarized_upto=28)
        out = skill_engine.render_analysis(analysis, max_chars=1_200)
        self.assertEqual(self.stub.calls, [], "scan already folded past the split; no new call needed")
        self.assertIn("PRIOR ROLLING SUMMARY", out)
        self.assertIn("[M029]", out)

    def test_scan_fold_is_extended_when_it_does_not_cover_enough(self):
        findings = [f"[M{i:03d}] " + "q" * 300 for i in range(30)]
        analysis = _analysis(findings, findings_summary="PRIOR", summarized_upto=3)
        skill_engine.render_analysis(analysis, max_chars=1_200)
        self.assertEqual(len(self.stub.calls), 1)
        sent = self.stub.calls[0]["user"]
        self.assertIn("PRIOR", sent, "the prior summary must be carried into the extended fold")
        self.assertNotIn("[M002]", sent, "findings already folded must not be folded twice")
        self.assertIn("[M003]", sent)

    # -- degraded and edge cases ------------------------------------------------------------------

    def test_empty_summary_is_flagged_as_incomplete(self):
        self.stub.reply = ""
        findings = [f"[M{i:03d}] " + "e" * 300 for i in range(20)]
        out = skill_engine.render_analysis(_analysis(findings), max_chars=1_000)
        self.assertIn("INCOMPLETE", out)
        self.assertIn("not the full exhibit", out)

    def test_single_oversized_finding_still_renders(self):
        findings = ["[M000] " + "b" * 50_000]
        out = skill_engine.render_analysis(_analysis(findings), max_chars=1_000)
        self.assertIn("[M000]", out)
        self.assertEqual(self.stub.calls, [])

    def test_no_findings(self):
        out = skill_engine.render_analysis(_analysis([]))
        self.assertIn("no relevant findings surfaced", out)

    def test_none_analysis_renders_nothing(self):
        self.assertEqual(skill_engine.render_analysis(None), "")

    # -- the GreatWhite_2023_0003 shape ------------------------------------------------------------

    def test_late_periods_survive_at_the_real_finding_size(self):
        """32 findings at the ~470 chars/finding measured on 0003, at the shipped budget."""
        periods = ["003.2022", "004.2022", "005.2022", "006.2022", "007.2022", "008.2022",
                   "009.2022", "010.2022", "012.2022"]
        findings = [
            f"Sheet1 row {i + 2}: Fiscal year/Period {periods[i % len(periods)]}, "
            f"Cost + OH {(i + 1) * 271.35:.2f}, " + "detail " * 55
            for i in range(32)
        ]
        out = skill_engine.render_analysis(_analysis(findings))
        self.assertEqual(self.stub.calls, [], "32 findings must fit the shipped budget uncompressed")
        for p in periods:
            self.assertIn(p, out, f"period {p} vanished from the synthesis block")
        self.assertIn("Sheet1 row 33", out, "the exhibit's last row must reach synthesis")


if __name__ == "__main__":
    unittest.main()
