"""Tests for the JADE ledger name harvest and ledger-derived person list.

Everything here runs offline. The workbook tests write a small .xlsx into a temp directory, so the
harvest opens a real file the same way it opens a corpus exhibit, and the one test that needs the light
tier replaces `otl.ledger_people.llm.structured` with a stub — no network call is made.

The behaviours worth guarding hardest:

* NOTHING IS DROPPED. The harvest has no row cap and no cap on distinct values on purpose. Every
  distinct value in the sheet has to come back, because a value left behind is a person left out of the
  whole rebuttal.
* The Employee column wins per ROW, and the line-item text is the fallback for exactly the rows where
  Employee says "Not assigned". That is the pattern the real ledgers use.
* A POB tab sharing the auditor's workbook must not contribute its passenger list once a real ledger
  tab has been found.
* The cap falls on the complete ledger-derived list after name deduplication.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from otl import ledger_people
from otl.agents._common import (MAX_DISPUTED_PEOPLE, cap_people, disputed_people,
                                max_disputed_people, set_max_disputed_people)
from otl.graph import route_from_jade_analyzer
from otl.workbook_manifest import build_manifest

# Four of the seven JADE signature tokens, so `detect_tab_type` fires and the tab comes back as a
# detected ledger rather than through the generous fallback.
_JADE_HEADER = ["Joint Venture(JV)", "Equity Group", "Cost Object WBS/CC (CO)", "Employee",
                "Line Item Text Orig", "Account", "Amount"]

# A POB report: a name column, no JADE signature. This is the tab that must stay out of the harvest.
_POB_HEADER = ["Date", "Personnel Name", "Company", "Location"]


def _row(employee: str, text: str, account: str = "A9820120", amount: float = 1000.0) -> list:
    return ["GW", "EG1", "C.WBS.1", employee, text, account, amount]


def _write_workbook(path: Path, sheets: dict[str, list[list]]) -> Path:
    from openpyxl import Workbook

    wb = Workbook()
    wb.remove(wb.active)
    for title, rows in sheets.items():
        ws = wb.create_sheet(title=title[:31])
        for row in rows:
            ws.append(row)
    wb.save(path)
    return path


def _ledger_rows() -> list[list]:
    """A ledger holding every case the harvest has to handle."""
    return [
        list(_JADE_HEADER),
        _row("OYLER, KEN", "2023-04 OFFSHORE SUPPORT 40H"),          # Employee wins
        _row("OYLER, KEN", "2023-05 OFFSHORE SUPPORT 38H"),          # duplicate Employee, one value
        _row("", "2023-04 SHEILA ADAMS PLATFORM ENGINEER"),          # blank Employee → text
        _row("Not assigned", "2023-05 N ADAMS OPERATIONS"),          # placeholder Employee → text
        _row("Unavailable", "2023-06 BLAKE CADO ENGINEERING"),       # placeholder Employee → text
        _row("-", "CONTRACT LABOR REBILL 2023-06"),                  # a value that names nobody
        _row("oyler, ken", "2023-07 OFFSHORE SUPPORT 12H"),          # same name, different casing
        _row("", ""),                                                # nothing at all → skipped
    ]


class ColumnMatchingTests(unittest.TestCase):
    """Which header label is the name column. Pure — no file needed."""

    def test_employee_id_is_not_a_name_column(self) -> None:
        self.assertTrue(ledger_people._is_employee_column("Employee"))
        self.assertTrue(ledger_people._is_employee_column("Employee Reporting Name"))
        self.assertFalse(ledger_people._is_employee_column("Employee ID"))
        self.assertFalse(ledger_people._is_employee_column("Employee Number"))

    def test_line_item_text_variants_match(self) -> None:
        self.assertTrue(ledger_people._is_text_column("Line Item Text Orig"))
        self.assertTrue(ledger_people._is_text_column("LINE ITEM TEXT"))
        self.assertTrue(ledger_people._is_text_column("Text"))
        self.assertFalse(ledger_people._is_text_column("Text Length"))

    def test_both_columns_are_reported_when_both_exist(self) -> None:
        # The choice is per ROW, so a tab that has both must report both.
        cols = ledger_people._person_columns(_JADE_HEADER)
        self.assertEqual(cols, {"employee": 3, "text": 4})


class HarvestTests(unittest.TestCase):
    """`find_ledger_tabs` + `harvest_person_values` against a real .xlsx."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _manifest(self, sheets: dict[str, list[list]], people=None) -> dict:
        path = _write_workbook(self.dir / "Labor Exception Support.xlsx", sheets)
        return build_manifest([str(path)], people=people or [])

    def test_finds_the_ledger_tab_and_its_columns(self) -> None:
        manifest = self._manifest({"Sheet1": _ledger_rows()})
        tabs = ledger_people.find_ledger_tabs(manifest)
        self.assertEqual(len(tabs), 1)
        entry = tabs[0]
        self.assertEqual(entry["tab"], "Sheet1")
        self.assertEqual(entry["detected_type"], "JADE")
        self.assertEqual(entry["header_row"], 0)
        self.assertEqual(entry["cols"], {"employee": 3, "text": 4})

    def test_employee_wins_per_row_and_text_is_the_fallback(self) -> None:
        manifest = self._manifest({"Sheet1": _ledger_rows()})
        entry = ledger_people.find_ledger_tabs(manifest)[0]
        values = ledger_people.harvest_person_values(
            entry["path"], entry["tab"], entry["header_row"], entry["cols"])
        self.assertEqual(values, [
            "OYLER, KEN",                              # Employee, and the duplicate row added nothing
            "2023-04 SHEILA ADAMS PLATFORM ENGINEER",  # blank Employee fell back to the text
            "2023-05 N ADAMS OPERATIONS",              # "Not assigned" fell back to the text
            "2023-06 BLAKE CADO ENGINEERING",           # "Unavailable" fell back to the text
            "CONTRACT LABOR REBILL 2023-06",           # kept — deciding it names nobody is the model's job
        ])

    def test_nothing_distinct_is_dropped(self) -> None:
        # 60 rows, each with its own name: every one has to come back. There is no cap here.
        rows = [list(_JADE_HEADER)] + [_row(f"PERSON {i:02d}", f"2023-04 SHIFT {i}") for i in range(60)]
        manifest = self._manifest({"Sheet1": rows})
        values, tabs_read = ledger_people.harvest(manifest)
        self.assertEqual(len(values), 60)
        self.assertEqual(values[0], "PERSON 00")
        self.assertEqual(values[-1], "PERSON 59")
        self.assertEqual(tabs_read, ["Labor Exception Support.xlsx!Sheet1"])

    def test_a_pob_tab_in_the_same_workbook_is_not_harvested(self) -> None:
        pob = [list(_POB_HEADER),
               ["2023-04-01", "SOMEONE ELSE", "Contractor", "Platform"],
               ["2023-04-02", "ANOTHER PASSENGER", "Contractor", "Platform"]]
        manifest = self._manifest({"JADE Detail": _ledger_rows(), "POB": pob})
        tabs = ledger_people.find_ledger_tabs(manifest)
        self.assertEqual([t["tab"] for t in tabs], ["JADE Detail"])
        values, _ = ledger_people.harvest(manifest)
        self.assertNotIn("ANOTHER PASSENGER", values)

    def test_a_renamed_exhibit_still_harvests_through_the_fallback(self) -> None:
        # No JADE signature at all, just a name column. The fallback pass is what keeps a hand-compiled
        # exhibit from failing the node's evidence gate.
        rows = [["Period", "Employee", "Cost", "Amount"],
                ["2023-04", "OYLER, KEN", "C.1", 1000.0]]
        manifest = self._manifest({"Labor": rows})
        tabs = ledger_people.find_ledger_tabs(manifest)
        self.assertEqual([t["tab"] for t in tabs], ["Labor"])
        self.assertEqual(tabs[0]["detected_type"], "")
        values, _ = ledger_people.harvest(manifest)
        self.assertEqual(values, ["OYLER, KEN"])

    def test_no_name_column_means_no_ledger_tab(self) -> None:
        # This is what fires the node's evidence gate.
        rows = [["Account", "Cost Object", "Amount"], ["A9820120", "C.1", 1000.0]]
        manifest = self._manifest({"Summary": rows})
        self.assertEqual(ledger_people.find_ledger_tabs(manifest), [])
        self.assertEqual(ledger_people.harvest(manifest), ([], []))

    def test_charge_totals_cover_the_complete_ledger_population_before_any_cap(self) -> None:
        rows = _ledger_rows() + [_row("WALSH, MICHAEL", "2023-06 OFFSHORE SUPPORT", amount=500.0)]
        manifest = self._manifest({"Sheet1": rows})

        totals = ledger_people.charged_amount_by_person(
            manifest, ["Ken Oyler", "Sheila Adams", "N Adams", "Michael Walsh"],
            warn=lambda _message: None,
        )

        self.assertEqual(totals, {
            "Ken Oyler": 3000.0,
            "Sheila Adams": 1000.0,
            "N Adams": 1000.0,
            "Michael Walsh": 500.0,
        })


class ExtractPeopleTests(unittest.TestCase):
    """The light-tier call, with the model replaced by a stub."""

    def setUp(self) -> None:
        self._real = ledger_people.llm.structured
        self.calls: list[list[str]] = []

    def tearDown(self) -> None:
        ledger_people.llm.structured = self._real

    def _stub(self, reply_for) -> None:
        """Replace the light-tier call. `reply_for` maps a chunk of values to the names to return."""
        def _structured(node, system, user, schema):
            values = [line[2:] for line in user.splitlines() if line.startswith("- ")]
            self.calls.append(values)
            return schema(people=reply_for(values))

        ledger_people.llm.structured = _structured

    def test_names_are_pulled_out_and_deduped_across_chunks(self) -> None:
        self._stub(lambda values: ["Ken Oyler", "Sheila Adams", "Ken Oyler"])
        names = ledger_people.extract_people("jade_people", ["OYLER, KEN", "2023-04 SHEILA ADAMS"])
        self.assertEqual(names, ["Ken Oyler", "Sheila Adams"])
        self.assertEqual(len(self.calls), 1)

    def test_intake_leads_are_merged_with_pos_suggestions_for_the_llm(self) -> None:
        seen_users: list[str] = []

        def _structured(node, system, user, schema):
            seen_users.append(user)
            return schema(people=["OYLER, KEN", "Sheila Adams"])

        ledger_people.llm.structured = _structured
        names = ledger_people.extract_people(
            "jade_people", ["OYLER, KEN"], intake_people=["Sheila Adams", "Sheila Adams"],
        )

        self.assertEqual(names, ["OYLER, KEN", "Sheila Adams"])
        self.assertEqual(len(seen_users), 1)
        self.assertIn("INTAKE LEAD PEOPLE:\n* Sheila Adams", seen_users[0])

    def test_every_value_reaches_a_call_when_the_list_is_long(self) -> None:
        # Long enough to need several chunks. The point of the chunking is coverage, not truncation.
        values = [f"2023-04 EMPLOYEE {i:03d} OFFSHORE SUPPORT ROTATION" for i in range(400)]
        self._stub(lambda chunk: [f"Person {chunk[0][8:]}"])
        ledger_people.extract_people("jade_people", values)
        self.assertGreater(len(self.calls), 1)
        self.assertEqual(sorted(v for call in self.calls for v in call), sorted(values))

    def test_a_failed_chunk_is_skipped_rather_than_failing_the_node(self) -> None:
        calls = {"n": 0}

        def _structured(node, system, user, schema):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("gateway timeout")
            return schema(people=["Sheila Adams"])

        ledger_people.llm.structured = _structured
        values = [f"2023-04 EMPLOYEE {i:03d} OFFSHORE SUPPORT ROTATION" for i in range(400)]
        names = ledger_people.extract_people("jade_people", values, warn=lambda m: None)
        self.assertEqual(names, ["Sheila Adams"])       # partial beats nothing

    def test_no_values_means_no_call_at_all(self) -> None:
        self._stub(lambda values: ["Should Not Happen"])
        self.assertEqual(ledger_people.extract_people("jade_people", []), [])
        self.assertEqual(self.calls, [])


class OnePersonOneEntryTests(unittest.TestCase):
    """The ledger-name dedupe has to recognise the same person written two ways."""

    def setUp(self) -> None:
        self._cap = max_disputed_people()
        set_max_disputed_people(0)          # no cap, so these are purely about the dedupe

    def tearDown(self) -> None:
        set_max_disputed_people(self._cap)

    def test_the_real_0003_names_collapse_to_one(self) -> None:
        kept, dropped = cap_people(
            ["Kenneth Oyler", "Oyler, Kenneth Willmann", "Oyler Kenneth Willmann"])
        self.assertEqual(kept, ["Kenneth Oyler"])
        self.assertEqual(dropped, [])

    def test_word_order_and_middle_names_do_not_make_a_second_person(self) -> None:
        kept, _ = cap_people(["Kenneth Oyler", "OYLER KENNETH", "Kenneth Willmann Oyler", "Oyler"])
        self.assertEqual(kept, ["Kenneth Oyler"])

    def test_a_generational_suffix_does_not_make_a_second_person(self) -> None:
        kept, _ = cap_people(["John Harrison", "Harrison, John Jr."])
        self.assertEqual(kept, ["John Harrison"])

    def test_two_people_sharing_a_surname_stay_separate(self) -> None:
        # Exception 0004 charges both Adamses. Collapsing them would silently drop one from the rebuttal.
        kept, _ = cap_people(["Adams, Nancy", "Adams, Sheila", "Nancy Adams", "Sheila W Adams"])
        self.assertEqual(kept, ["Adams, Nancy", "Adams, Sheila"])

    def test_an_ambiguous_bare_surname_is_kept_rather_than_guessed(self) -> None:
        # "Adams" fits both Adamses. Picking one could drop a third Adams entirely, so it stands on its
        # own and gets examined.
        kept, _ = cap_people(["Adams, Nancy", "Adams, Sheila", "Adams"])
        self.assertEqual(kept, ["Adams, Nancy", "Adams, Sheila", "Adams"])

    def test_an_unambiguous_bare_surname_collapses(self) -> None:
        kept, _ = cap_people(["Adams, Nancy", "Adams"])
        self.assertEqual(kept, ["Adams, Nancy"])

    def test_initials_are_not_their_own_person(self) -> None:
        kept, _ = cap_people(["Kenneth Oyler", "K. Oyler", "Oyler, K W"])
        self.assertEqual(kept, ["Kenneth Oyler"])

    def test_a_name_written_only_in_initials_still_survives(self) -> None:
        # No tokens survive normalization, so the exact-string key is all there is to go on. Keeping it
        # is the safe failure: an unexamined person is worse than a redundant one.
        kept, _ = cap_people(["K O", "K O", "J P"])
        self.assertEqual(kept, ["K O", "J P"])

    def test_a_collapsed_variant_does_not_consume_a_cap_slot(self) -> None:
        set_max_disputed_people(2)
        kept, dropped = cap_people(
            ["Kenneth Oyler", "Oyler, Kenneth Willmann", "Sheila Adams", "Nancy Adams"])
        self.assertEqual(kept, ["Kenneth Oyler", "Sheila Adams"])
        self.assertEqual(dropped, ["Nancy Adams"])

    def test_a_variant_of_a_dropped_name_does_not_come_back(self) -> None:
        set_max_disputed_people(1)
        kept, dropped = cap_people(["Kenneth Oyler", "Sheila Adams", "Adams, Sheila Marie"])
        self.assertEqual(kept, ["Kenneth Oyler"])
        self.assertEqual(dropped, ["Sheila Adams"])


class CascadeTests(unittest.TestCase):
    """`disputed_people` — the single read point every specialist after jade goes through."""

    def setUp(self) -> None:
        self._cap = max_disputed_people()

    def tearDown(self) -> None:
        set_max_disputed_people(self._cap)

    def test_default_has_no_person_cap(self) -> None:
        self.assertEqual(MAX_DISPUTED_PEOPLE, 0)
        set_max_disputed_people(MAX_DISPUTED_PEOPLE)
        names = [f"Person{n:03d}" for n in range(51)]
        state = {"jade_analyzer": {"people_found_in_ledger": names}}
        self.assertEqual(disputed_people(state), names)

    def test_the_jade_population_is_the_only_source(self) -> None:
        set_max_disputed_people(5)
        state = {"jade_analyzer": {"people_found_in_ledger": ["Kenneth Oyler", "Sheila Adams"]}}
        self.assertEqual(disputed_people(state), ["Kenneth Oyler", "Sheila Adams"])

    def test_no_population_exists_before_jade_has_run(self) -> None:
        set_max_disputed_people(5)
        self.assertEqual(disputed_people({}), [])

    def test_the_cap_still_bounds_the_ledger_list(self) -> None:
        # The bound has to hold however the channel got populated — a checkpoint from a run with a
        # larger cap must not widen this one.
        set_max_disputed_people(2)
        state = {"jade_analyzer": {"people_found_in_ledger": ["A A", "B B", "C C", "D D"]}}
        self.assertEqual(disputed_people(state), ["A A", "B B"])

    def test_an_empty_jade_population_stays_empty(self) -> None:
        set_max_disputed_people(5)
        state = {"jade_analyzer": {"people_found_in_ledger": []}}
        self.assertEqual(disputed_people(state), [])


class GateRoutingTests(unittest.TestCase):
    """`route_from_jade_analyzer` — the edge the no-ledger gate needs. It has to fail OPEN."""

    def test_a_normal_run_goes_to_the_hub(self) -> None:
        # A normal run is not operationally blocked, so it must reach the orchestrator.
        self.assertEqual(route_from_jade_analyzer({}), "orchestrator")
        self.assertEqual(route_from_jade_analyzer({"analysis_blocked": False}), "orchestrator")

    def test_only_an_explicit_false_reaches_the_drafter(self) -> None:
        self.assertEqual(route_from_jade_analyzer({"analysis_blocked": True}), "drafter")


if __name__ == "__main__":
    unittest.main()
