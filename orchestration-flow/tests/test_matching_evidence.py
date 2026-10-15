from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from otl.agents import matching as matching_agent
from otl.matching_evidence import (CfrMappingCandidate, MatchingDecision, MatchingPersonFinding,
                                   MatchingPersonPacket, analyze_packets, build_shards,
                                   candidates_for_person, mapping_rows)


MAPPING = ("/Users/harshakolachina/Documents/Projects/Shell JV Audit/Data Files/OTL Information/"
           "staff_listing_with_CFR_mapping 1.xlsx")


class CfrWorkbookTests(unittest.TestCase):
    def test_real_mapping_workbook_is_deduplicated(self):
        if not os.path.exists(MAPPING):
            self.skipTest("CFR workbook unavailable")
        rows = mapping_rows(MAPPING)
        self.assertGreater(len(rows), 800)
        self.assertLess(len(rows), 10_000)
        self.assertTrue(any(row.row_count > 100 for row in rows))

    def test_exact_profile_candidates_preserve_cfr_text(self):
        rows = (
            CfrMappingCandidate(candidate_id="1", job_profile_description="Wells Engineer JG4",
                                job_title="Staff Wells Engineer", cfr_primary="30 CFR 250",
                                rationale="Well delivery mapping."),
            CfrMappingCandidate(candidate_id="2", job_profile_description="Business Support JG5",
                                job_title="Administrator"),
        )
        candidates = candidates_for_person(
            positions=["Staff Wells Engineer"], profiles=["Wells Engineer JG4"], rows=rows)
        self.assertEqual([row.candidate_id for row in candidates], ["1"])
        self.assertEqual(candidates[0].cfr_primary, "30 CFR 250")

    def test_packets_remain_atomic_when_sharded(self):
        packets = [MatchingPersonPacket(evidence_id=f"matching:{i:04d}", person=f"Person {i}")
                   for i in range(50)]
        shards, reasons = build_shards(packets)
        self.assertFalse(reasons)
        self.assertEqual([p.evidence_id for shard in shards for p in shard],
                         [p.evidence_id for p in packets])

    def test_structured_mapper_cannot_create_chargeability_result(self):
        candidate_a = CfrMappingCandidate(candidate_id="a", job_profile_description="Profile A")
        candidate_b = CfrMappingCandidate(candidate_id="b", job_profile_description="Profile B")
        packet = MatchingPersonPacket(
            evidence_id="matching:0001", person="Jane Smith", positions=["Engineer"],
            candidates=[candidate_a, candidate_b])
        decision = MatchingDecision(findings=[MatchingPersonFinding(
            person="Jane Smith", evidence_ids_consumed=[packet.evidence_id],
            selected_candidate_ids=["a"], mapping_status="matched", rationale="Role/profile aligned.")])
        with patch("otl.matching_evidence.llm.structured", return_value=decision):
            findings, reasons, shards = analyze_packets([packet])
        self.assertFalse(reasons)
        self.assertEqual(shards, 1)
        self.assertEqual(findings[0].selected_candidate_ids, ["a"])


class MatchingNodeFlowTests(unittest.TestCase):
    def _state(self):
        return {
            "case": {"case_id": "case-1", "asset": "Great White"},
            "otl_information_dir": "/otl",
            "job_title": {
                "positions_by_person": {"Jane Smith": ["Staff Wells Engineer"]},
                "selected_job_profiles_by_person": {"Jane Smith": ["Wells Engineer JG4"]},
                "identified_job_data": {"Wells Engineer JG4": "Designs and executes wells."},
                "identified_job_descriptions": {"Wells Engineer JG4": "Well engineering duties."},
            },
        }

    def test_normal_structured_path_never_constructs_code_executor(self):
        candidate = CfrMappingCandidate(
            candidate_id="a", job_profile_description="Wells Engineer JG4",
            cfr_primary="30 CFR 250", subpart_labels="Subpart D")
        packet = MatchingPersonPacket(
            evidence_id="matching:0001", person="Jane Smith",
            positions=["Staff Wells Engineer"], selected_job_profiles=["Wells Engineer JG4"],
            candidates=[candidate])
        finding = MatchingPersonFinding(
            person="Jane Smith", evidence_ids_consumed=[packet.evidence_id],
            selected_candidate_ids=["a"], mapping_status="matched")
        with patch.object(matching_agent, "_cfr_mapping", return_value=MAPPING), \
                patch.object(matching_agent, "build_manifest", return_value={}), \
                patch.object(matching_agent, "prepare_packets", return_value=[packet]), \
                patch.object(matching_agent, "analyze_packets", return_value=([finding], {}, 0)), \
                patch.object(matching_agent, "CodeExecutor") as executor:
            update = matching_agent.matching_node(self._state())
        executor.assert_not_called()
        sub = update["matching"]
        self.assertEqual(sub.matching_analysis_mode, "structured")
        self.assertEqual(sub.cfr_mapping_status_by_person["Jane Smith"], "matched")
        self.assertEqual(sub.cfr_primary_by_person["Jane Smith"], ["30 CFR 250"])
        self.assertEqual(sub.cfr_copas_resolution_by_person["Jane Smith"], "chargeable")
        self.assertEqual(sub.cfr_copas_resolution, "chargeable")
        self.assertEqual(sub.identified_cfr_category, "")
        self.assertEqual(sub.cfr_clause, "")

    def test_fallback_receives_only_unresolved_people(self):
        packets = [
            MatchingPersonPacket(evidence_id="matching:0001", person="Resolved Person",
                                 candidates=[CfrMappingCandidate(
                                     candidate_id="a", job_profile_description="Profile A")]),
            MatchingPersonPacket(evidence_id="matching:0002", person="Unresolved Person",
                                 candidates=[CfrMappingCandidate(candidate_id="b", job_profile_description="B"),
                                             CfrMappingCandidate(candidate_id="c", job_profile_description="C")]),
        ]
        resolved = MatchingPersonFinding(
            person="Resolved Person", evidence_ids_consumed=["matching:0001"],
            selected_candidate_ids=["a"], mapping_status="matched")
        loop = SimpleNamespace(steps=1, status="finished", result={
            "cfr_mapping_status_by_person": {"Unresolved Person": "matched"},
            "matched_cfr_profiles_by_person": {"Unresolved Person": ["Profile B"]},
        })
        state = self._state()
        state["job_title"]["positions_by_person"]["Resolved Person"] = ["A"]
        state["job_title"]["positions_by_person"]["Unresolved Person"] = ["B"]
        with patch.object(matching_agent, "_cfr_mapping", return_value=MAPPING), \
                patch.object(matching_agent, "build_manifest", return_value={}), \
                patch.object(matching_agent, "prepare_packets", return_value=packets), \
                patch.object(matching_agent, "analyze_packets", return_value=(
                    [resolved], {"Unresolved Person": ["multiple close CFR candidates"]}, 1)), \
                patch.object(matching_agent, "_fallback", return_value=loop) as fallback:
            matching_agent.matching_node(state)
        self.assertEqual(fallback.call_args.kwargs["people"], ["Unresolved Person"])


if __name__ == "__main__":
    unittest.main()
