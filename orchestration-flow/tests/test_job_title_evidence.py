from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from otl import ipt_registry
from otl.agents import job_title as job_title_agent
from otl.job_title_evidence import (JobTitlePersonFinding, JobTitlePersonPacket, PreparedJobTitleEvidence,
                                    JobTitleDecision, StaffTitlePeriod, analyze_packets, build_shards,
                                    catalog_candidates, catalog_rows, exempt_status,
                                    prepare_evidence, title_periods)


CATALOG = "/Users/harshakolachina/Documents/Projects/Shell JV Audit/Data Files/OTL Information/Jobs_Catalogue.xlsx"


class IPTRegistryTests(unittest.TestCase):
    def test_registry_is_scoped_by_audit_year_and_asset(self):
        match = ipt_registry.lookup(audit_year="2025", asset="Whale", person="Lee White")
        self.assertIsNotNone(match)
        self.assertIn("Offshore Installation", match["registry_title"])
        self.assertIsNone(ipt_registry.lookup(audit_year="2024", asset="Vito", person="Lee White"))

    def test_registry_absence_is_non_ipt(self):
        status, evidence = ipt_registry.status(
            audit_year="2025", asset="Whale", person="Person Not In Chart")
        self.assertEqual(status, "non_ipt")
        self.assertIsNone(evidence)

    def test_matrix_context_does_not_become_a_title(self):
        match = ipt_registry.lookup(audit_year="2025", asset="Vito", person="Oyler, Kenneth")
        self.assertIsNotNone(match)
        self.assertIsNone(match["registry_title"])
        self.assertIn("VA007 Bypass Drilling", match["chart_context"])

    def test_registry_titles_feed_exempt_role_matching(self):
        match = ipt_registry.lookup(audit_year="2025", asset="Whale", person="Lee White")
        self.assertEqual(exempt_status(positions=[], profiles=[], ipt_match=match), "exempt")
        self.assertEqual(exempt_status(
            positions=["Lead Wells Engineer"], profiles=[], ipt_match=None), "exempt")


class StaffTitlePeriodTests(unittest.TestCase):
    def _index(self):
        entry = {
            "name": "staff.xlsx", "tab": "Staff", "columns": {
                "person_key": {"Employee ID": 0}, "position": {"Position": 1},
                "job_profile": {"Job Profile": 2}, "reports_to": {"Manager ID": 3},
                "manager_name": {"Manager Name": 4}, "period": {"Month": 5},
            },
            "row_data": {"Cado, Blake": [
                {"_row": 1, "Employee ID": "215209", "Position": "Project Engineer",
                 "Job Profile": "", "Manager ID": "10", "Manager Name": "Manager One",
                 "Month": "January", "Master Cost Center Key": "101013",
                 "Master Cost Center Name": "Mars Ops"},
                {"_row": 2, "Employee ID": "215209", "Position": "Project Engineer",
                 "Job Profile": "", "Manager ID": "10", "Manager Name": "Manager One",
                 "Month": "February", "Master Cost Center Key": "101013",
                 "Master Cost Center Name": "Mars Ops"},
                {"_row": 3, "Employee ID": "215209", "Position": "Construction Lead",
                 "Job Profile": "", "Manager ID": "10", "Manager Name": "Manager One",
                 "Month": "March", "Master Cost Center Key": "101013",
                 "Master Cost Center Name": "Mars Ops"},
            ]},
        }
        return {"sources": [{"role": "staff", "entries": [entry], "selected": {
            "Cado, Blake": [{"name_in_file": "Cado, Blake", "confidence": "high"}]
        }}]}

    def test_monthly_rows_collapse_into_distinct_title_periods(self):
        periods, reasons, _source = title_periods(
            self._index(), ["Cado, Blake"],
            {"jade_analyzer": {"ledger_rows": [{"person": "Cado, Blake", "employee_id": "215209"}]}},
        )
        self.assertFalse(reasons)
        self.assertEqual(len(periods["Cado, Blake"]), 2)
        self.assertEqual(periods["Cado, Blake"][0].months, ["January", "February"])
        self.assertEqual(periods["Cado, Blake"][1].months, ["March"])


class CatalogueAndShardingTests(unittest.TestCase):
    def test_active_catalogue_sheet_is_indexed(self):
        if not os.path.exists(CATALOG):
            self.skipTest("Jobs Catalogue unavailable")
        rows = catalog_rows(CATALOG)
        self.assertGreater(len(rows), 3000)
        candidates = catalog_candidates(
            StaffTitlePeriod(adopted_name="A", position="Project Engineer"), rows)
        self.assertGreaterEqual(len(candidates), 5)
        self.assertLessEqual(len(candidates), 8)

    def test_person_packets_remain_atomic_when_sharded(self):
        packets = [JobTitlePersonPacket(
            evidence_id=f"job-title:{i:04d}", person=f"Person {i}", identity_status="resolved",
            staff_title_periods=[StaffTitlePeriod(adopted_name=f"Person {i}", position="Engineer")],
        ) for i in range(50)]
        shards, reasons = build_shards(packets)
        self.assertFalse(reasons)
        ids = [packet.evidence_id for shard in shards for packet in shard]
        self.assertEqual(ids, [packet.evidence_id for packet in packets])

    def test_conflicting_title_periods_are_forced_to_unclear(self):
        packet = JobTitlePersonPacket(
            evidence_id="job-title:0001", person="Mixed Person", identity_status="resolved",
            staff_title_periods=[
                StaffTitlePeriod(adopted_name="Mixed Person", position="Engineer"),
                StaffTitlePeriod(adopted_name="Mixed Person", position="Administrative Assistant"),
            ],
            catalog_candidates_by_period={"period-01": [], "period-02": []},
        )
        decision = JobTitleDecision(findings=[JobTitlePersonFinding(
            person="Mixed Person", evidence_ids_consumed=[packet.evidence_id],
            adopted_positions=["Engineer", "Administrative Assistant"],
            technical_status="technical", technical_rationale="Mixed duties.",
            job_profile_status="matched",
            technical_status_by_period={"period-01": "technical", "period-02": "non_technical"},
            selected_job_profiles_by_period={"period-01": [], "period-02": []},
        )])
        with patch("otl.job_title_evidence.llm.structured", return_value=decision):
            findings, reasons, _shards = analyze_packets([packet])
        self.assertEqual(findings[0].technical_status, "unclear")
        self.assertTrue(findings[0].needs_raw_review)
        self.assertIn("technical status conflicts", reasons["Mixed Person"][0])


class JobTitleNodeFlowTests(unittest.TestCase):
    def test_normal_structured_path_never_constructs_code_executor(self):
        packet = JobTitlePersonPacket(
            evidence_id="job-title:0001", person="Cado, Blake", identity_status="resolved",
            staff_title_periods=[StaffTitlePeriod(adopted_name="Cado, Blake", position="Project Engineer")],
            ipt_registry_match={"name": "Cado, Blake", "ipt": True, "status": "ipt"},
        )
        prepared = PreparedJobTitleEvidence(packets=[packet])
        finding = JobTitlePersonFinding(
            person="Cado, Blake", evidence_ids_consumed=[packet.evidence_id],
            adopted_positions=["Project Engineer"], selected_job_profiles=[],
            technical_status="technical", technical_rationale="Engineering duties are technical.",
            job_profile_status="matched",
        )
        state = {
            "case": {"case_id": "case-1", "year": "2025", "asset": "Whale"},
            "jade_analyzer": {"people_found_in_ledger": ["Cado, Blake"]},
            "common_reports_dir": "/reports", "otl_information_dir": "/otl",
        }
        with patch.object(job_title_agent, "resolve_staff_listing", return_value="/reports/staff.xlsx"), \
                patch.object(job_title_agent, "_jobs_catalogue", return_value=CATALOG), \
                patch.object(job_title_agent, "build_manifest", return_value={}), \
                patch.object(job_title_agent, "build_person_index", return_value={}), \
                patch.object(job_title_agent, "prepare_evidence", return_value=prepared), \
                patch.object(job_title_agent, "analyze_packets", return_value=([finding], {}, 1)), \
                patch.object(job_title_agent, "CodeExecutor") as executor:
            update = job_title_agent.job_title_node(state)
        executor.assert_not_called()
        self.assertEqual(update["job_title"].job_title_analysis_mode, "structured")
        self.assertEqual(update["job_title"].role_status_by_person["Cado, Blake"], "technical")
        self.assertEqual(update["job_title"].ipt_status_by_person["Cado, Blake"], "ipt")

    def test_fallback_receives_only_unresolved_people(self):
        packets = [
            JobTitlePersonPacket(
                evidence_id="job-title:0001", person="Resolved Person", identity_status="resolved",
                staff_title_periods=[StaffTitlePeriod(adopted_name="Resolved Person", position="Engineer")],
            ),
            JobTitlePersonPacket(
                evidence_id="job-title:0002", person="Unresolved Person", identity_status="resolved",
                staff_title_periods=[StaffTitlePeriod(adopted_name="Unresolved Person", position="Coordinator")],
            ),
        ]
        prepared = PreparedJobTitleEvidence(packets=packets)
        finding = JobTitlePersonFinding(
            person="Resolved Person", evidence_ids_consumed=["job-title:0001"],
            adopted_positions=["Engineer"], technical_status="technical",
            technical_rationale="Engineering duties are technical.", job_profile_status="matched",
        )
        loop = SimpleNamespace(
            steps=2, status="finished", result={
                "role_status_by_person": {"Unresolved Person": "non_technical"},
                "technical_rationale_by_person": {"Unresolved Person": "Administrative coordination."},
                "selected_job_profiles_by_person": {"Unresolved Person": ["Business Support"]},
                "job_profile_status_by_person": {"Unresolved Person": "matched"},
                "positions_by_person": {"Unresolved Person": ["Coordinator"]},
            },
        )
        state = {
            "case": {"case_id": "case-1", "year": "2025", "asset": "Whale"},
            "jade_analyzer": {"people_found_in_ledger": ["Resolved Person", "Unresolved Person"]},
            "common_reports_dir": "/reports", "otl_information_dir": "/otl",
        }
        with patch.object(job_title_agent, "resolve_staff_listing", return_value="/reports/staff.xlsx"), \
                patch.object(job_title_agent, "_jobs_catalogue", return_value=CATALOG), \
                patch.object(job_title_agent, "build_manifest", return_value={}), \
                patch.object(job_title_agent, "build_person_index", return_value={}), \
                patch.object(job_title_agent, "prepare_evidence", return_value=prepared), \
                patch.object(job_title_agent, "analyze_packets", return_value=(
                    [finding], {"Unresolved Person": ["multiple close profiles"]}, 1)), \
                patch.object(job_title_agent, "_fallback", return_value=loop) as fallback:
            update = job_title_agent.job_title_node(state)
        self.assertEqual(fallback.call_args.kwargs["people"], ["Unresolved Person"])
        self.assertEqual(update["job_title"].role_status_by_person["Resolved Person"], "technical")
        self.assertEqual(update["job_title"].role_status_by_person["Unresolved Person"], "non_technical")


if __name__ == "__main__":
    unittest.main()
