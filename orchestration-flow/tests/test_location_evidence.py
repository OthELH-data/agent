"""Offline tests for the one-pass JSON location evidence pipeline."""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from openpyxl import Workbook

import otl  # noqa: F401  -- bootstraps the shared extractor package path
from shared.token_accounting import MODEL_INPUT_LIMITS
from otl import location_evidence
from otl.agents import location as location_agent
from otl.state import LocationState
from otl.workbook_manifest import build_manifest
from otl.person_index import classify_columns


def _workbook(path: Path, header: list, rows: list[list], *, sheet: str) -> str:
    wb = Workbook()
    ws = wb.active
    ws.title = sheet
    ws.append(header)
    for row in rows:
        ws.append(row)
    wb.save(path)
    return str(path)


class AzureContextTests(unittest.TestCase):
    def test_all_azure_tiers_have_configured_input_tokens(self) -> None:
        self.assertEqual(MODEL_INPUT_LIMITS["gpt-5.6-sol"], 1_050_000)
        self.assertEqual(MODEL_INPUT_LIMITS["gpt-5.6-terra"], 1_050_000)
        self.assertEqual(MODEL_INPUT_LIMITS["gpt-5.6-luna"], 1_050_000)


class LocationEvidenceTests(unittest.TestCase):
    def test_comma_ordered_people_use_the_actual_surname(self) -> None:
        self.assertEqual(location_evidence._person_parts("Cado, Blake"), ("cado", ["blake"]))
        self.assertEqual(
            location_evidence._candidate_people("BLAKE CADO", ["Cado, Blake", "Hudson, Blake"]),
            ["Cado, Blake"],
        )

    def test_full_timestamp_in_year_labeled_column_is_preserved(self) -> None:
        header = ["END TIME - Year", "END TIME - Month", "END TIME - Day"]
        value, precision = location_evidence._component_date(
            [datetime(2024, 11, 19, 16, 39), None, None],
            location_evidence._header_map(header), "END TIME",
        )
        self.assertEqual((value, precision), ("2024-11-19", "day"))

    def test_compact_starttime_and_endtime_headers_are_supported(self) -> None:
        header = ["StartTime", "EndTime", "AssetDescription", "TripItemName", "Origin", "Destination"]
        event = location_evidence._event_for(
            "heli", ["1/4/2019 7:44:00 AM", "1/4/2019 8:55:00 AM", "Q4000",
                     "Nathan Leighton", "PHI HOUMA", "Q4000"], header,
            workbook="2019.xlsx", sheet="Trip Data", inferred_year=2019,
        )
        self.assertEqual(event["start_date"], "2019-01-04")
        self.assertEqual(event["end_date"], "2019-01-04")
        self.assertEqual(event["asset"], "Q4000")

    def test_trip_item_name_is_recognized_as_helicopter_subject(self) -> None:
        classified = classify_columns([
            "TripItemName", "AssetDescription", "StartTime", "EndTime", "Origin", "Destination",
        ])
        self.assertEqual(classified["subject"]["labels"], ["TripItemName"])

    def test_route_and_company_aliases_are_normalized(self) -> None:
        header = ["Passenger Name", "Asset Name", "From Location", "To Location", "Flight Number",
                  "Employer", "Flight Date"]
        event = location_evidence._event_for(
            "heli", ["JANE SMITH", "MARS", "PHI HOUMA", "MARS", "F100", "SHELL", "03/15/2023"],
            header, workbook="flight.xlsx", sheet="Data", inferred_year=2023,
        )
        self.assertEqual(event["asset"], "MARS")
        self.assertEqual((event["origin"], event["destination"]), ("PHI HOUMA", "MARS"))
        self.assertEqual((event["start_date"], event["end_date"]), ("2023-03-15", "2023-03-15"))
        self.assertEqual(event["company"], "SHELL")

    def test_pob_location_role_company_and_date_aliases_are_normalized(self) -> None:
        header = ["Crew Member Name", "Installation", "Employer", "Job Title", "Check In", "Check Out"]
        event = location_evidence._event_for(
            "pob", ["JANE SMITH", "URSA", "SHELL", "ENGINEER", "03/01/2023", "03/15/2023"],
            header, workbook="pob.xlsx", sheet="POB", inferred_year=2023,
        )
        self.assertEqual(event["location"], "URSA")
        self.assertEqual(event["company"], "SHELL")
        self.assertEqual(event["position"], "ENGINEER")
        self.assertEqual((event["arrival_date"], event["departure_date"]),
                         ("2023-03-01", "2023-03-15"))

    def test_month_day_components_use_source_year(self) -> None:
        header = ["START TIME - Month", "START TIME - Day",
                  "END TIME - Month", "END TIME - Day"]
        event = location_evidence._event_for(
            "heli", ["January", 1, "January", 2], header,
            workbook="2020.xlsx", sheet="Sheet1", inferred_year=2020,
        )
        self.assertEqual(event["start_date"], "2020-01-01")
        self.assertEqual(event["end_date"], "2020-01-02")
        self.assertEqual(event["date_precision"], "day")

    def test_partial_component_subsets_preserve_available_precision(self) -> None:
        month, precision = location_evidence._component_date(
            ["March"], location_evidence._header_map(["END TIME - Month"]),
            "END TIME", inferred_year=2021,
        )
        self.assertEqual((month, precision), ("2021-03", "month"))
        year, precision = location_evidence._component_date(
            [2021], location_evidence._header_map(["END TIME - Year"]), "END TIME")
        self.assertEqual((year, precision), ("2021", "year"))

    def test_source_year_prefers_filename_then_audit_year_fallback(self) -> None:
        self.assertEqual(location_evidence._source_year("2020 Helicopter.xlsx", "2021"), 2020)
        self.assertEqual(location_evidence._source_year("Helicopter Export.xlsx", "2022"), 2021)

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_pob_timezone_suffix_is_normalized(self) -> None:
        self.assertEqual(location_evidence._date_text(
            "10/31/2022 15:50 Central Daylight Time"), "2022-10-31")

    def test_one_pass_artifact_keeps_selected_evidence_and_excludes_unrelated_people(self) -> None:
        pob = _workbook(
            self.root / "pob.xlsx",
            ["First Name", "Last Name", "Location", "Company", "Position", "Arrival Date",
             "Departure Date"],
            [
                ["KENNETH", "OYLER", "PROTEUS", "SHELL", "ENGINEER", "10/31/2022", "11/11/2022"],
                ["OTHER", "PERSON", "MARS", "OTHER", "TECH", "10/01/2022", "10/02/2022"],
            ],
            sheet="PersonnelData",
        )
        heli = _workbook(
            self.root / "heli.xlsx",
            ["NAME", "COMPANY", "ASSET", "TRIP ID", "ORIGIN", "DESTINATION",
             "END TIME - Year", "END TIME - Month", "END TIME - Day"],
            [
                ["KENNETH OYLER", "SHELL", "PROTEUS", "T1", "HOUMA", "PROTEUS", 2022, 10, 31],
                ["UNRELATED PERSON", "OTHER", "MARS", "T2", "A", "B", 2022, 10, 31],
            ],
            sheet="Helicopter Log",
        )
        manifest = build_manifest([[pob], [heli]])

        def select_stub(_node, role, entries, people, warn=None):
            values = [v for entry in entries for v in entry.get("values") or []]
            selected = [{"name_in_file": v, "confidence": "high", "why": "test match"}
                        for v in values if "oyler" in v.lower()]
            return ({"Kenneth Oyler": selected} if selected else {}), 1

        scratch = self.root / "scratch"
        scratch.mkdir()
        with patch.object(location_evidence, "select_names", side_effect=select_stub), patch.object(
                location_evidence, "_scratch_dir", return_value=scratch):
            result = location_evidence.extract_location_evidence(
                run_id="test", audit_year="2023", charged_asset="PROTEUS",
                people=["Kenneth Oyler"], pob_paths=[pob], heli_paths=[heli], manifest=manifest,
            )

        self.assertEqual(result.rows_scanned, {"pob": 2, "heli": 2})
        self.assertFalse(result.unresolved_reasons)
        artifact = json.loads(Path(result.artifact_path).read_text())
        person = artifact["people"]["Kenneth Oyler"]
        self.assertEqual(person["pob"]["raw_row_count"], 1)
        self.assertEqual(person["heli"]["raw_row_count"], 1)
        self.assertEqual(person["pob"]["intervals"][0]["arrival_date"], "2022-10-31")
        self.assertEqual(person["heli"]["events"][0]["date_precision"], "day")
        self.assertEqual(person["heli"]["events"][0]["end_date"], "2022-10-31")
        self.assertNotIn("UNRELATED", json.dumps(artifact))

    def test_sharding_preserves_each_evidence_id_once(self) -> None:
        artifact = {
            "schema_version": 1,
            "case": {},
            "identity_selection": {},
            "warnings": [],
            "people": {
                "A Person": {
                    "pob": {"intervals": []},
                    "heli": {"events": [
                        {"evidence_id": f"heli:a:{i:04d}", "asset": "X", "detail": "z" * 100}
                        for i in range(20)
                    ]},
                }
            },
        }
        with patch.object(location_evidence.llm, "estimate_text_tokens",
                          side_effect=lambda text: max(1, len(text) // 4)):
            shards = location_evidence.build_shards(artifact, target_tokens=500, max_tokens=700)
        ids = [item["event"]["evidence_id"] for shard in shards for item in shard["events"]]
        self.assertGreater(len(shards), 1)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {f"heli:a:{i:04d}" for i in range(20)})

    def test_name_sheet_without_location_columns_is_an_unsupported_schema(self) -> None:
        path = _workbook(self.root / "bad.xlsx", ["NAME", "COMMENT"],
                         [["KENNETH OYLER", "nothing useful"]], sheet="Sheet1")
        manifest = build_manifest([path])
        with self.assertRaisesRegex(ValueError, "no usable name-bearing heli sheet"):
            location_evidence._scan_source("heli", path, ["Kenneth Oyler"], manifest, [])


class LocationNodeTests(unittest.TestCase):
    def _state(self) -> dict:
        return {
            "run_id": "run-1",
            "case": {"case_id": "case-1", "year": "2025", "asset": "PROTEUS"},
            "jade_analyzer": {"people_found_in_ledger": ["Kenneth Oyler"]},
            "common_reports_dir": "/reports/2025",
            "sandbox_read_roots": ["/reports/2025"],
        }

    def test_normal_path_never_constructs_code_executor(self) -> None:
        artifact = {"case": {}, "people": {}, "evidence_ids": [], "extraction_statistics": {}}
        extracted = location_evidence.ExtractionResult(
            artifact=artifact, artifact_path="/scratch/location_evidence.json",
            unresolved_reasons=[], rows_scanned={"pob": 10, "heli": 20},
        )
        decision = location_evidence.LocationDecision(result=LocationState(
            nl_summary="Kenneth Oyler was supported by the reviewed evidence as present on PROTEUS.",
            found_in_pob={"Kenneth Oyler": True},
            found_in_heli_logs={"Kenneth Oyler": True},
        ))
        with patch.object(location_agent, "_common_reports_files",
                          return_value=("/reports/2025", ["/reports/pob.xlsx"], ["/reports/heli.xlsx"])), \
                patch.object(location_agent, "build_manifest", return_value={}), \
                patch.object(location_agent, "extract_location_evidence", return_value=extracted), \
                patch.object(location_agent, "_structured_decision", return_value=(decision, [], 1)), \
                patch.object(location_agent, "artifact_tokens", return_value=100), \
                patch.object(location_agent, "CodeExecutor") as executor:
            update = location_agent.location_node(self._state())
        executor.assert_not_called()
        self.assertEqual(update["agent_status"]["location"]["status"], "objective_met")
        self.assertTrue(update["location"].found_in_pob["Kenneth Oyler"])

    def test_escalation_keeps_full_executor_and_uses_twelve_steps(self) -> None:
        loop = SimpleNamespace(status="finished", result={"nl_summary": "The location question was resolved from the available evidence."}, steps=2,
                               last_observation="")
        executor_instance = object()
        with patch.object(location_agent, "CodeExecutor", return_value=executor_instance) as constructor, \
                patch.object(location_agent, "run_agent_loop", return_value=loop) as runner:
            got_loop, _sub = location_agent._escalate(
                self._state(), manifest={}, artifact_path="/scratch/location_evidence.json",
                reasons=["unresolved_identity:heli:Kenneth Oyler"], people=["Kenneth Oyler"],
                charged_asset="PROTEUS", pob_reports=["/reports/pob.xlsx"],
                heli_logs=["/reports/heli.xlsx"], common_reports_dir="/reports/2025",
            )
        self.assertIs(got_loop, loop)
        constructor.assert_called_once_with("run-1", "location", read_roots=["/reports/2025"], manifest={})
        self.assertEqual(runner.call_args.kwargs["max_steps"], 12)
        self.assertIs(runner.call_args.kwargs["executor"], executor_instance)

    def test_250k_threshold_uses_single_call_then_shards(self) -> None:
        artifact = {"evidence_ids": [], "people": {}}
        one = location_evidence.LocationDecision(result=LocationState())
        with patch.object(location_agent, "artifact_tokens", return_value=249_999), \
                patch.object(location_agent, "_single_decision", return_value=(one, [])) as single, \
                patch.object(location_agent, "_shard_findings") as shards:
            decision, reasons, count = location_agent._structured_decision(self._state(), artifact)
        self.assertIs(decision, one)
        self.assertFalse(reasons)
        self.assertEqual(count, 1)
        single.assert_called_once()
        shards.assert_not_called()

        finding = location_evidence.LocationShardFinding(shard_id="s1")
        with patch.object(location_agent, "artifact_tokens", return_value=250_001), \
                patch.object(location_agent, "_shard_findings", return_value=([finding], [])) as shards, \
                patch.object(location_agent, "_reduce_findings", return_value=(one, [])) as reducer:
            decision, reasons, count = location_agent._structured_decision(self._state(), artifact)
        self.assertIs(decision, one)
        self.assertFalse(reasons)
        self.assertEqual(count, 1)
        shards.assert_called_once()
        reducer.assert_called_once()

    def test_incomplete_presence_maps_require_escalation(self) -> None:
        decision = location_evidence.LocationDecision(result=LocationState(
            found_in_pob={"Kenneth Oyler": True}, found_in_heli_logs={}))
        reasons = location_agent._decision_completeness(decision, ["Kenneth Oyler"])
        self.assertEqual(len(reasons), 1)
        self.assertIn("found_in_heli_logs", reasons[0])

    def test_questioned_period_uses_auditor_month_codes_then_jade_dates(self) -> None:
        start, end = location_agent._questioned_period({
            "auditor_narrative": "Questioned service months were 0322 through 0622.",
        })
        self.assertEqual((start, end), ("2022-03-01", "2022-06-30"))

        start, end = location_agent._questioned_period({
            "jade_analyzer": {"ledger_rows": [
                {"entry_date_sl": "2023-04-15"}, {"entry_date_sl": "2023-06-03"},
            ]}
        })
        self.assertEqual((start, end), ("2023-04-15", "2023-06-03"))


class DeterministicLocationStatusTests(unittest.TestCase):
    @staticmethod
    def _artifact(*, selected_pob=None, selected_heli=None, pob_events=None, heli_events=None,
                  unresolved=None, pob_coverage=("2023-01-01", "2023-12-31"),
                  heli_coverage=("2023-01-01", "2023-12-31")):
        person = "Jane Smith"
        return {
            "case": {"charge_period_start": "2023-03-01", "charge_period_end": "2023-03-31"},
            "identity_selection": {person: {
                "pob": {"selected_names": selected_pob or [], "confidence": "high"},
                "heli": {"selected_names": selected_heli or [], "confidence": "high"},
            }},
            "people": {person: {
                "pob": {"intervals": pob_events or []},
                "heli": {"events": heli_events or []},
            }},
            "source_coverage": {
                "pob": {"start_date": pob_coverage[0], "end_date": pob_coverage[1],
                        "dated_rows": 10, "successful_sources": 1},
                "heli": {"start_date": heli_coverage[0], "end_date": heli_coverage[1],
                         "dated_rows": 10, "successful_sources": 1},
            },
            "unresolved_reasons": unresolved or [],
        }

    def test_definitive_absence_from_both_sources_is_offsite(self):
        statuses, reasons = location_evidence.classify_location_statuses(self._artifact())
        self.assertEqual(statuses["Jane Smith"], "offsite")
        self.assertIn("no definitive POB or helicopter record", reasons["Jane Smith"])

    def test_overlapping_definitive_record_is_onsite(self):
        artifact = self._artifact(
            selected_pob=["SMITH, JANE"],
            pob_events=[{"arrival_date": "2023-03-10", "departure_date": "2023-03-12"}],
        )
        statuses, _reasons = location_evidence.classify_location_statuses(artifact)
        self.assertEqual(statuses["Jane Smith"], "onsite")

    def test_end_only_dated_event_can_establish_onsite(self):
        artifact = self._artifact(
            selected_heli=["JANE SMITH"],
            heli_events=[{"start_date": "", "end_date": "2023-03-10", "date_precision": "day"}],
        )
        statuses, _reasons = location_evidence.classify_location_statuses(artifact)
        self.assertEqual(statuses["Jane Smith"], "onsite")

    def test_partial_month_event_requires_full_month_inside_questioned_period(self):
        artifact = self._artifact(
            selected_heli=["JANE SMITH"],
            heli_events=[{"start_date": "", "end_date": "2023-03", "date_precision": "month"}],
        )
        statuses, _reasons = location_evidence.classify_location_statuses(artifact)
        self.assertEqual(statuses["Jane Smith"], "onsite")
        artifact["case"] = {"charge_period_start": "2023-03-10", "charge_period_end": "2023-03-12"}
        statuses, _reasons = location_evidence.classify_location_statuses(artifact)
        self.assertEqual(statuses["Jane Smith"], "unresolved")

    def test_multiple_close_names_are_unresolved(self):
        artifact = self._artifact(selected_pob=["SMITH, JANE", "SMITH, JANET"])
        statuses, _reasons = location_evidence.classify_location_statuses(artifact)
        self.assertEqual(statuses["Jane Smith"], "unresolved")

    def test_exact_reordered_aliases_are_not_treated_as_competing_people(self):
        artifact = self._artifact(
            selected_pob=["SMITH, JANE", "JANE SMITH"],
            pob_events=[{"arrival_date": "2023-03-10", "departure_date": "2023-03-12"}],
        )
        statuses, _reasons = location_evidence.classify_location_statuses(artifact)
        self.assertEqual(statuses["Jane Smith"], "onsite")

    def test_incomplete_source_period_is_unresolved(self):
        artifact = self._artifact(heli_coverage=("2023-04-01", "2023-12-31"))
        statuses, _reasons = location_evidence.classify_location_statuses(artifact)
        self.assertEqual(statuses["Jane Smith"], "unresolved")


if __name__ == "__main__":
    unittest.main()
