"""Tests for the person index — the one-pass read of the big corpus workbooks.

Everything here runs offline. The harvest tests write small `.xlsx` files into a temp directory, so the
read opens a real file the same way it opens a corpus workbook, and every test that needs the med tier
replaces `otl.person_index.llm.structured` with a stub — no network call is made.

The behaviours worth guarding hardest:

* NOTHING IS DROPPED IN THE READ. No row cap, no cap on distinct names, and every distinct value has to
  land in exactly one selection chunk. A value left behind is a person left out of the whole rebuttal.
* COLUMNS ARE READ, NEVER ASSUMED. The 2021 staff listing is three schemas in one workbook, and the
  labels its tabs use for the person key and the reports-to pointer disagree. Classification is an
  allow-list of exact labels, so `Headcount Name` is not a person and `Employee ID` is not a name.
* A TAB WITH NO NAME COLUMN, OR NO HEADER ROW, IS SKIPPED AND LISTED. Skipping quietly is what would let
  an agent conclude a file holds nothing.
* THE MODEL MAY ONLY CHOOSE AMONG THE VALUES IT WAS SHOWN. A returned spelling that is not in the file is
  dropped and warned, because a hallucinated spelling matches no row and reads as "person absent".
* ABSENCE IS A FINDING. A person nothing was selected for comes back with the size of the pool that
  shares their surname — the WILLIAMS case from the 0033 trace.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from otl import llm_adapter, person_index
from otl.code_exec import CodeExecutor
from otl.workbook_manifest import build_manifest

# The 2021 staff listing's three tab families, headers as that workbook writes them. These are the
# schemas that make the whole "return the columns" requirement necessary.
_STAFF_REPORTING = ["Employee ID", "Employee Reporting Name", "Manager ID", "Manager Name",
                    "Position", "Job Profile", "Month", "Headcount Name", "Sup Org Name"]
_STAFF_WORKER = ["Employee ID", "Worker", "Supervisory Organization - ID", "Position", "Month",
                 "Master Cost Center Name"]
_STAFF_LEGAL = ["Employee Number", "Employee Name", "Supervisor Number", "Supervisor Name",
                "Position Text", "Month", "Employee Type"]

# A POB report: the name lives in two columns that only mean a person together.
_POB_HEADER = ["First Name", "Last Name", "Location", "Company", "Arrival Date", "Departure Date",
               "Essential"]


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


def _staff_rows(header: list, names: list[str], key_start: int = 100) -> list[list]:
    """A staff tab: the header, then one row per name filled positionally."""
    rows = [list(header)]
    for i, name in enumerate(names):
        row = [f"E{key_start + i}", name]
        row += [f"{header[j]} v{i}" for j in range(2, len(header))]
        rows.append(row)
    return rows


def _pob_rows(pairs: list[tuple[str, str]]) -> list[list]:
    """A POB tab: the header, then one row per (first, last) pair."""
    rows = [list(_POB_HEADER)]
    for i, (first, last) in enumerate(pairs):
        rows.append([first, last, "MARS", "SHELL", f"2023-04-{i + 1:02d}",
                     f"2023-04-{i + 15:02d}", "Y"])
    return rows


class ColumnClassificationTests(unittest.TestCase):
    """Which column is which. Pure — no file needed."""

    def test_the_three_2021_families_classify_differently(self) -> None:
        # This is the whole reason the columns come back as key-value pairs rather than being assumed.
        a = person_index.classify_columns(_STAFF_REPORTING)
        b = person_index.classify_columns(_STAFF_WORKER)
        c = person_index.classify_columns(_STAFF_LEGAL)
        self.assertEqual(a["subject"]["labels"], ["Employee Reporting Name"])
        self.assertEqual(b["subject"]["labels"], ["Worker"])
        self.assertEqual(c["subject"]["labels"], ["Employee Name"])
        self.assertEqual(list(a["columns"]["person_key"]), ["Employee ID"])
        self.assertEqual(list(c["columns"]["person_key"]), ["Employee Number"])
        self.assertEqual(list(a["columns"]["reports_to"]), ["Manager ID"])
        self.assertEqual(list(b["columns"]["reports_to"]), ["Supervisory Organization - ID"])
        self.assertEqual(list(c["columns"]["reports_to"]), ["Supervisor Number"])
        self.assertEqual(list(a["columns"]["manager_name"]), ["Manager Name"])
        self.assertEqual(list(c["columns"]["manager_name"]), ["Supervisor Name"])

    def test_job_profile_absence_is_recorded_not_guessed(self) -> None:
        # `Job Profile` is on the 2022 staff tab and gone from 2023 and 2024. An empty role is what lets
        # the prompt block say "not present" instead of leaving the agent to read and find out.
        with_profile = person_index.classify_columns(_STAFF_REPORTING)
        without = person_index.classify_columns(_STAFF_WORKER)
        self.assertEqual(list(with_profile["columns"]["job_profile"]), ["Job Profile"])
        self.assertIn("job_profile", without["columns"])
        self.assertEqual(without["columns"]["job_profile"], {})

    def test_name_like_labels_that_are_not_people(self) -> None:
        # Every one of these contains "name", "employee" or "worker", and none is a person.
        reporting = person_index.classify_columns(_STAFF_REPORTING)
        legal = person_index.classify_columns(_STAFF_LEGAL)
        self.assertIn("Headcount Name", reporting["unclassified_nameish"])
        self.assertIn("Employee Type", legal["unclassified_nameish"])
        # Org units get a role of their own, so they are visibly classified rather than reported as
        # name-like leftovers an agent might mistake for a second name column.
        self.assertEqual(list(reporting["columns"]["org"]), ["Sup Org Name"])
        self.assertEqual(list(person_index.classify_columns(_STAFF_WORKER)["columns"]["org"]),
                         ["Master Cost Center Name"])

    def test_manager_email_and_position_text_are_not_manager_names(self) -> None:
        # `manager_name` holds a person. A manager's email address and the text of their position do
        # not, and matching on "contains manager" would take both.
        header = ["Worker", "Manager Email", "Manager Position Text", "Manager Name"]
        got = person_index.classify_columns(header)
        self.assertEqual(list(got["columns"]["manager_name"]), ["Manager Name"])
        claimed = {label for cols in got["columns"].values() for label in cols}
        self.assertNotIn("Manager Email", claimed)
        self.assertNotIn("Manager Position Text", claimed)
        # `Position Text` IS the position role, so the guard is exactness, not a blanket refusal.
        self.assertEqual(list(person_index.classify_columns(
            ["Worker", "Position Text"])["columns"]["position"]), ["Position Text"])

    def test_an_id_label_is_never_a_name_column(self) -> None:
        self.assertTrue(person_index._is_id_label("Employee ID"))
        self.assertTrue(person_index._is_id_label("Employee Number"))
        self.assertTrue(person_index._is_id_label("Supervisor Number"))
        self.assertFalse(person_index._is_id_label("Employee Reporting Name"))
        self.assertFalse(person_index._is_id_label("Worker"))
        # `Employee ID` sits first in the header and still does not win the subject slot.
        got = person_index.classify_columns(["Employee ID", "Employee Reporting Name"])
        self.assertEqual(got["subject"]["labels"], ["Employee Reporting Name"])

    def test_a_first_last_pair_is_the_subject_but_a_lone_first_name_is_not(self) -> None:
        pair = person_index.classify_columns(_POB_HEADER)
        self.assertEqual(pair["subject"]["kind"], "first_last")
        self.assertEqual(pair["subject"]["labels"], ["Last Name", "First Name"])
        self.assertEqual(pair["subject"]["indices"], [1, 0])
        self.assertIsNone(person_index.classify_columns(["First Name", "Location"])["subject"])

    def test_a_combined_column_wins_over_a_pair_and_the_pair_is_kept(self) -> None:
        header = ["First Name", "Last Name", "Employee Reporting Name", "Location"]
        got = person_index.classify_columns(header)
        self.assertEqual(got["subject"]["kind"], "combined")
        self.assertEqual(got["subject"]["labels"], ["Employee Reporting Name"])
        self.assertEqual(got["columns"]["name_parts"], {"First Name": 0, "Last Name": 1})

    def test_several_subject_columns_on_one_tab_are_all_harvested_in_precedence_order(self) -> None:
        header = ["Worker", "Employee Reporting Name", "Name", "Position"]
        got = person_index.classify_columns(header)
        self.assertEqual(got["subject"]["kind"], "combined")
        self.assertEqual(got["subject"]["labels"], ["Employee Reporting Name", "Worker", "Name"])
        self.assertEqual(got["subject"]["indices"], [1, 0, 2])

    def test_a_tab_with_no_subject_column_has_none(self) -> None:
        # The 2025 staff workbook's junk `Sheet1`, whose "header row" is really a data row.
        header = ["Caleb Wells", "January", "Subsea Engineering Manager", "Maggio, Adam"]
        self.assertIsNone(person_index.classify_columns(header)["subject"])

    def test_the_tier_is_light(self) -> None:
        self.assertEqual(llm_adapter.tier_for("person_index"), "light")


class SkipGuardTests(unittest.TestCase):
    """The two hard guards, tested straight against a manifest workbook entry."""

    def test_a_tab_with_no_header_row_is_skipped_and_listed(self) -> None:
        wb_entry = {"name": "heli.xlsx", "path": "/nowhere/heli.xlsx",
                    "shapes": [{"id": "s1", "header_row": None, "header": [], "tabs": ["Raw"]}]}
        entries, skipped = person_index._entries_for_workbook(wb_entry)
        self.assertEqual(entries, [])
        self.assertEqual(skipped, [{"name": "heli.xlsx", "tab": "Raw", "reason": "no header row"}])

    def test_every_tab_of_a_no_name_shape_is_listed(self) -> None:
        wb_entry = {"name": "pob.xlsx", "path": "/nowhere/pob.xlsx",
                    "shapes": [{"id": "s1", "header_row": 0,
                                "header": ["Region", "Headcount"],
                                "tabs": ["GOM", "UNDEFINED", "Summaries"]}]}
        entries, skipped = person_index._entries_for_workbook(wb_entry)
        self.assertEqual(entries, [])
        self.assertEqual([s["tab"] for s in skipped], ["GOM", "UNDEFINED", "Summaries"])
        self.assertEqual({s["reason"] for s in skipped}, {"no name column"})

    def test_one_entry_per_tab_of_a_shared_shape(self) -> None:
        wb_entry = {"name": "staff.xlsx", "path": "/nowhere/staff.xlsx",
                    "shapes": [{"id": "s1", "header_row": 2, "header": list(_STAFF_WORKER),
                                "tabs": ["IG2", "P&T2", "UP Function2"]}]}
        entries, skipped = person_index._entries_for_workbook(wb_entry)
        self.assertEqual(skipped, [])
        self.assertEqual([e["tab"] for e in entries], ["IG2", "P&T2", "UP Function2"])
        self.assertEqual({e["header_row"] for e in entries}, {2})


class HarvestTests(unittest.TestCase):
    """Phase 1 against real `.xlsx` files. Deterministic — no model involved."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _harvest(self, sheets: dict[str, list[list]], name: str = "staff.xlsx"):
        path = _write_workbook(self.dir / name, sheets)
        manifest = build_manifest([str(path)])
        return person_index.harvest_names(manifest, [str(path)])

    def test_nothing_distinct_is_dropped(self) -> None:
        # 300 rows, each its own name. There is no cap here, and there must not be one.
        names = [f"PERSON {i:03d}" for i in range(300)]
        entries, skipped = self._harvest({"Staff Listing": _staff_rows(_STAFF_REPORTING, names)})
        self.assertEqual(skipped, [])
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["values"], names)
        self.assertEqual(entries[0]["rows"], 300)

    def test_placeholders_blanks_and_repeats_are_skipped(self) -> None:
        names = ["OYLER, KEN", "oyler, ken", "", "Not assigned", "-", "ADAMS, SHEILA"]
        entries, _ = self._harvest({"Staff Listing": _staff_rows(_STAFF_REPORTING, names)})
        # Same name in different casing is one value, and the file's first spelling is the one kept.
        self.assertEqual(entries[0]["values"], ["OYLER, KEN", "ADAMS, SHEILA"])

    def test_a_first_last_pair_is_joined_last_first(self) -> None:
        entries, _ = self._harvest(
            {"PersonnelData": _pob_rows([("JOHN", "HARRISON"), ("KYLE", "PRICHARD"), ("", "SOLO")])},
            name="pob.xlsx")
        self.assertEqual(entries[0]["subject"]["kind"], "first_last")
        self.assertEqual(entries[0]["values"], ["HARRISON, JOHN", "PRICHARD, KYLE", "SOLO"])

    def test_a_banner_row_partway_down_is_skipped(self) -> None:
        rows = _pob_rows([("JOHN", "HARRISON")])
        rows.append(["Personnel On Board"] * len(_POB_HEADER))     # the banner that recurs mid-tab
        rows += _pob_rows([("SUSAN", "PARK")])[1:]
        entries, _ = self._harvest({"PersonnelData": rows}, name="pob.xlsx")
        self.assertEqual(entries[0]["values"], ["HARRISON, JOHN", "PARK, SUSAN"])

    def test_a_no_name_tab_in_the_same_workbook_is_skipped_and_listed(self) -> None:
        entries, skipped = self._harvest({
            "PersonnelData": _pob_rows([("JOHN", "HARRISON")]),
            "Summaries": [["Region", "Headcount"], ["GOM", 412]],
        }, name="pob.xlsx")
        self.assertEqual([e["tab"] for e in entries], ["PersonnelData"])
        self.assertEqual([(s["tab"], s["reason"]) for s in skipped],
                         [("Summaries", "no name column")])

    def test_three_subject_spellings_in_one_workbook_are_three_entries(self) -> None:
        # The 2021 case, in miniature. One person appears on all three tabs under three column names.
        entries, skipped = self._harvest({
            "UP Business3": _staff_rows(_STAFF_REPORTING, ["OYLER, KEN", "ADAMS, SHEILA"]),
            "UP Business2": _staff_rows(_STAFF_WORKER, ["OYLER, KEN", "PARK, SUSAN"], key_start=200),
            "UP Business": _staff_rows(_STAFF_LEGAL, ["OYLER, KENNETH"], key_start=300),
        })
        self.assertEqual(skipped, [])
        self.assertEqual(len(entries), 3)
        self.assertEqual({e["subject"]["labels"][0] for e in entries},
                         {"Employee Reporting Name", "Worker", "Employee Name"})
        # And they disagree on the reports-to pointer, which is the point.
        self.assertEqual({lbl for e in entries for lbl in e["columns"]["reports_to"]},
                         {"Manager ID", "Supervisory Organization - ID", "Supervisor Number"})


class SelectNamesTests(unittest.TestCase):
    """Phase 2 with the light tier replaced by a stub."""

    def setUp(self) -> None:
        self._real = person_index.llm.structured
        self.calls: list[list[str]] = []
        self.nodes: list[str] = []
        self.prompts: list[str] = []
        self.warnings: list[str] = []
        self._env = {k: os.environ.get(k)
                     for k in ("OTL_NAME_SELECT_CHUNK_CHARS", "OTL_NAME_SELECT_JOBS")}

    def tearDown(self) -> None:
        person_index.llm.structured = self._real
        for key, value in self._env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _force_chunking(self, jobs: str = "1") -> None:
        """Shrink the chunk budget to its floor so a few hundred values need several calls.

        The real budget is 1,000,000 characters — about 250k tokens by the rough estimate — and a fixture
        small enough to read in a test fits in one call. The floor is 2,000, which is the smallest the module
        will accept.
        """
        os.environ["OTL_NAME_SELECT_CHUNK_CHARS"] = "2000"
        os.environ["OTL_NAME_SELECT_JOBS"] = jobs

    def _entry(self, values: list[str], kind: str = "combined") -> dict:
        subject = ({"kind": "combined", "labels": ["Employee Reporting Name"], "indices": [1]}
                   if kind == "combined"
                   else {"kind": "first_last", "labels": ["Last Name", "First Name"],
                         "indices": [1, 0]})
        return {"path": "/nowhere/staff.xlsx", "name": "staff.xlsx", "tab": "Staff Listing",
                "header_row": 0, "header": list(_STAFF_REPORTING), "shape": "s1",
                "subject": subject, "columns": {}, "unclassified_nameish": [],
                "rows": len(values), "values": list(values), "row_data": {}, "row_counts": {}}

    def _stub(self, reply_for) -> None:
        """Replace the med-tier call. `reply_for` maps one chunk's values to `(person, value)` pairs."""
        def _structured(node, system, user, schema, **kwargs):
            # The prompt lists the file's values first and the disputed people second, both as "- x"
            # lines, so split on the sentence that separates them.
            head = user.split("These are the people under dispute")[0]
            values = [line[2:] for line in head.splitlines() if line.startswith("- ")]
            self.nodes.append(node)
            self.calls.append(values)
            self.prompts.append(user)
            matches = [{"person": p, "name_in_file": v, "confidence": "high", "why": "surname and "
                        "given name match"} for p, v in reply_for(values)]
            return schema(matches=matches)

        person_index.llm.structured = _structured

    def test_a_selection_comes_back_with_the_file_spelling(self) -> None:
        self._stub(lambda values: [("Oyler, Ken", "OYLER, KENNETH")])
        entry = self._entry(["OYLER, KENNETH", "PARK, SUSAN"])
        selected, chunks = person_index.select_names(
            "person_index", "staff", [entry], ["Oyler, Ken"], warn=self.warnings.append)
        self.assertEqual(chunks, 1)
        self.assertEqual(selected["Oyler, Ken"][0]["name_in_file"], "OYLER, KENNETH")
        self.assertEqual(self.warnings, [])

    def test_all_source_name_selection_uses_light_tier_node(self) -> None:
        self._stub(lambda values: [])
        entry = self._entry(["OYLER, KENNETH"])
        person_index.select_names("person_index", "pob", [entry], ["Oyler, Ken"],
                                  warn=self.warnings.append)
        person_index.select_names("person_index", "heli", [entry], ["Oyler, Ken"],
                                  warn=self.warnings.append)
        person_index.select_names("person_index", "staff", [entry], ["Oyler, Ken"],
                                  warn=self.warnings.append)
        self.assertEqual(self.nodes, ["person_index", "person_index", "person_index"])

    def test_every_value_reaches_exactly_one_chunk(self) -> None:
        # Coverage is the whole point of chunking: no value may be skipped and none may be asked twice.
        self._force_chunking(jobs="4")
        values = [f"EMPLOYEE {i:04d}, ROTATION OFFSHORE SUPPORT" for i in range(400)]
        self._stub(lambda chunk: [])
        _selected, chunks = person_index.select_names(
            "person_index", "staff", [self._entry(values)], ["Oyler, Ken"],
            warn=self.warnings.append)
        self.assertGreater(chunks, 1)
        self.assertEqual(len(self.calls), chunks)
        seen = [v for call in self.calls for v in call]
        self.assertEqual(sorted(seen), sorted(values))
        self.assertEqual(len(seen), len(set(seen)))

    def test_the_universe_is_unioned_across_tabs_so_one_pass_serves_them_all(self) -> None:
        # The 2021 case: three tabs, three subject spellings, largely the same workforce. Asking each
        # tab separately would pay for the same names three times.
        self._stub(lambda values: [("Oyler, Ken", "OYLER, KEN")] if "OYLER, KEN" in values else [])
        entries = [self._entry(["OYLER, KEN", "PARK, SUSAN"]),
                   self._entry(["OYLER, KEN", "ADAMS, SHEILA"]),
                   self._entry(["OYLER, KEN"])]
        selected, chunks = person_index.select_names(
            "person_index", "staff", entries, ["Oyler, Ken"], warn=self.warnings.append)
        self.assertEqual(chunks, 0)
        self.assertEqual(len(self.calls), 0)
        self.assertEqual(len(selected["Oyler, Ken"]), 1)

    def test_a_spelling_the_file_does_not_hold_is_dropped_and_warned(self) -> None:
        # A hallucinated spelling matches no row, so it would read as "person absent from the file".
        self._stub(lambda values: [("Oyler, Ken", "OYLER, KENNY-BOY")])
        selected, _ = person_index.select_names(
            "person_index", "staff", [self._entry(["OYLER, KENNETH"])], ["Oyler, Ken"],
            warn=self.warnings.append)
        self.assertEqual(selected, {})
        self.assertEqual(len(self.warnings), 1)
        self.assertIn("not in the file's name list", self.warnings[0])

    def test_a_person_we_did_not_ask_about_is_dropped_and_warned(self) -> None:
        self._stub(lambda values: [("Someone Else", "OYLER, KENNETH")])
        selected, _ = person_index.select_names(
            "person_index", "staff", [self._entry(["OYLER, KENNETH"])], ["Oyler, Ken"],
            warn=self.warnings.append)
        self.assertEqual(selected, {})
        self.assertIn("not one of the disputed people", self.warnings[0])

    def test_a_reformatted_person_name_still_resolves(self) -> None:
        # The model echoing "Ken Oyler" back for "Oyler, Ken" is a formatting difference, not a new person.
        self._stub(lambda values: [("Ken Oyler", "OYLER, KENNETH")])
        selected, _ = person_index.select_names(
            "person_index", "staff", [self._entry(["OYLER, KENNETH"])], ["Oyler, Ken"],
            warn=self.warnings.append)
        self.assertIn("Oyler, Ken", selected)
        self.assertEqual(self.warnings, [])

    def test_an_ambiguous_surname_is_refused_rather_than_guessed(self) -> None:
        # Two Adamses are two people; picking one would put the wrong person's rows in the rebuttal.
        self._stub(lambda values: [("Adams", "ADAMS, SHEILA")])
        selected, _ = person_index.select_names(
            "person_index", "staff", [self._entry(["ADAMS, SHEILA"])],
            ["Adams, Sheila", "Adams, Nathan"], warn=self.warnings.append)
        self.assertEqual(list(selected), ["Adams, Sheila"])
        self.assertEqual(selected["Adams, Sheila"][0]["name_in_file"], "ADAMS, SHEILA")

    def test_one_failing_chunk_is_warned_and_the_others_still_contribute(self) -> None:
        # A partial selection still beats a per-person scan of a 224,000-row workbook, so one dead call
        # must not take the node down with it. Serial, so "the first chunk" is unambiguous.
        self._force_chunking(jobs="1")
        values = [f"EMPLOYEE {i:04d}, ROTATION OFFSHORE SUPPORT" for i in range(400)]
        values.append("OYLER, KENNETH")
        state = {"failed": False}

        def _structured(node, system, user, schema, **kwargs):
            head = user.split("These are the people under dispute")[0]
            chunk = [line[2:] for line in head.splitlines() if line.startswith("- ")]
            if not state["failed"]:
                state["failed"] = True
                raise RuntimeError("gateway said no")
            hits = [{"person": "Oyler, Ken", "name_in_file": "OYLER, KENNETH", "confidence": "high",
                     "why": ""}] if "OYLER, KENNETH" in chunk else []
            return schema(matches=hits)

        person_index.llm.structured = _structured
        selected, chunks = person_index.select_names(
            "person_index", "staff", [self._entry(values)], ["Oyler, Ken"],
            warn=self.warnings.append)
        self.assertGreater(chunks, 1)
        self.assertEqual(selected["Oyler, Ken"][0]["name_in_file"], "OYLER, KENNETH")
        self.assertEqual(len(self.warnings), 1)
        self.assertIn("name selection over", self.warnings[0])

    def test_no_people_means_no_call(self) -> None:
        self._stub(lambda values: [])
        selected, chunks = person_index.select_names(
            "person_index", "staff", [self._entry(["OYLER, KENNETH"])], [],
            warn=self.warnings.append)
        self.assertEqual((selected, chunks), ({}, 0))
        self.assertEqual(self.calls, [])

    def test_the_prompt_states_the_column_shape_and_the_kind_of_file(self) -> None:
        # The model cannot judge "PARK, SUSAN" against "Susan Park" without being told the file writes
        # the surname first.
        self._stub(lambda values: [])
        person_index.select_names("person_index", "pob", [self._entry(["PARK, SUSAN"], "first_last")],
                                  ["Park, Suzie"], warn=self.warnings.append)
        prompt = self.prompts[-1]
        self.assertIn('joined here as "LAST, FIRST"', prompt)
        self.assertIn("`Last Name`", prompt)
        self.assertIn("personnel-on-board", prompt)
        self.assertIn("- PARK, SUSAN", prompt)
        self.assertIn("- Park, Suzie", prompt)

    def test_a_mixed_source_says_the_word_order_is_inconsistent(self) -> None:
        # `location` indexes the POB report (two columns) and the helicopter manifest (one) as separate
        # sources, but a single source can still mix shapes, and then neither rule holds alone.
        self._stub(lambda values: [])
        person_index.select_names(
            "person_index", "staff",
            [self._entry(["PARK, SUSAN"], "first_last"), self._entry(["OYLER, KENNETH"])],
            ["Park, Suzie"], warn=self.warnings.append)
        self.assertIn("Some tabs hold the whole name in one column", self.prompts[-1])


class RowPullTests(unittest.TestCase):
    """Phase 3 against a real `.xlsx`."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _entries(self, sheets: dict[str, list[list]], name: str = "staff.xlsx") -> list[dict]:
        path = _write_workbook(self.dir / name, sheets)
        entries, _ = person_index.harvest_names(build_manifest([str(path)]), [str(path)])
        return entries

    def test_rows_are_keyed_by_column_label_and_match_only_the_exact_spelling(self) -> None:
        entries = self._entries({"Staff Listing": _staff_rows(
            _STAFF_REPORTING, ["OYLER, KENNETH", "OYLER, KENDRA"])})
        selected = {"Oyler, Ken": [{"name_in_file": "OYLER, KENNETH"}]}
        person_index.pull_rows(entries, selected, warn=lambda m: None)
        rows = entries[0]["row_data"]["OYLER, KENNETH"]
        self.assertEqual(len(rows), 1)
        self.assertNotIn("OYLER, KENDRA", entries[0]["row_data"])
        self.assertEqual(rows[0]["Employee Reporting Name"], "OYLER, KENNETH")
        self.assertEqual(rows[0]["Employee ID"], "E100")
        self.assertEqual(rows[0]["_tab"], "Staff Listing")

    def test_rows_come_from_every_tab_that_holds_the_spelling(self) -> None:
        entries = self._entries({
            "UP Business3": _staff_rows(_STAFF_REPORTING, ["OYLER, KEN"]),
            "UP Business2": _staff_rows(_STAFF_WORKER, ["OYLER, KEN"], key_start=200),
        })
        person_index.pull_rows(entries, {"Oyler, Ken": [{"name_in_file": "OYLER, KEN"}]},
                               warn=lambda m: None)
        holding = {e["tab"] for e in entries if e["row_counts"].get("OYLER, KEN")}
        self.assertEqual(holding, {"UP Business3", "UP Business2"})

    def test_truncation_keeps_the_true_count_and_says_so(self) -> None:
        entries = self._entries({"Staff Listing": _staff_rows(
            _STAFF_REPORTING, ["OYLER, KEN"] * 12)})
        warnings: list[str] = []
        real = person_index._MAX_ROWS_PER_NAME
        person_index._MAX_ROWS_PER_NAME = 5
        try:
            person_index.pull_rows(entries, {"Oyler, Ken": [{"name_in_file": "OYLER, KEN"}]},
                                   warn=warnings.append)
        finally:
            person_index._MAX_ROWS_PER_NAME = real
        self.assertEqual(len(entries[0]["row_data"]["OYLER, KEN"]), 5)
        self.assertEqual(entries[0]["row_counts"]["OYLER, KEN"], 12)     # the TRUE count
        self.assertTrue(any("12 rows" in w for w in warnings))

    def test_nothing_selected_means_no_second_pass(self) -> None:
        entries = self._entries({"Staff Listing": _staff_rows(_STAFF_REPORTING, ["OYLER, KEN"])})
        person_index.pull_rows(entries, {}, warn=lambda m: None)
        self.assertEqual(entries[0]["row_data"], {})


class BuildIndexTests(unittest.TestCase):
    """`build_person_index` end to end, with the med tier stubbed."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self._real = person_index.llm.structured
        self.calls = 0
        person_index._INDEX_CACHE.clear()
        self.warnings: list[str] = []

    def tearDown(self) -> None:
        person_index.llm.structured = self._real
        person_index._INDEX_CACHE.clear()
        os.environ.pop("OTL_PERSON_INDEX", None)
        self.tmp.cleanup()

    def _stub(self, pairs: list[tuple[str, str]]) -> None:
        def _structured(node, system, user, schema, **kwargs):
            self.calls += 1
            head = user.split("These are the people under dispute")[0]
            chunk = [line[2:] for line in head.splitlines() if line.startswith("- ")]
            return schema(matches=[{"person": p, "name_in_file": v, "confidence": "high", "why": ""}
                                   for p, v in pairs if v in chunk])

        person_index.llm.structured = _structured

    def _staff_file(self) -> str:
        path = _write_workbook(self.dir / "Consolidated Staff Listing.xlsx", {
            "Staff Listing 2023": _staff_rows(
                _STAFF_REPORTING,
                ["OYLER, KENNETH", "WILLIAMS, JASON", "WILLIAMS, MARK", "PARK, SUSAN"]),
            "Summary": [["Region", "Headcount"], ["GOM", 412]],
        })
        return str(path)

    def test_a_full_build_records_the_selection_the_rows_and_the_tabs(self) -> None:
        path = self._staff_file()
        self._stub([("Oyler, Ken", "OYLER, KENNETH")])
        index = person_index.build_person_index(
            [("staff", [path])], ["Oyler, Ken", "Williams, Donell"],
            build_manifest([path]), warn=self.warnings.append)
        self.assertTrue(index["enabled"])
        source = index["sources"][0]
        self.assertEqual(source["role"], "staff")
        self.assertEqual(source["universe"], 4)
        self.assertEqual(source["chunks"], 1)
        match = source["selected"]["Oyler, Ken"][0]
        self.assertEqual(match["name_in_file"], "OYLER, KENNETH")
        self.assertEqual(match["rows"], 1)
        self.assertEqual(match["tabs"], ["Staff Listing 2023"])
        self.assertEqual([(s["tab"], s["reason"]) for s in source["skipped_tabs"]],
                         [("Summary", "no name column")])

    def test_absence_comes_back_with_the_size_of_the_surname_pool(self) -> None:
        # The WILLIAMS case from the 0033 trace: two values share the surname and neither is this person.
        path = self._staff_file()
        self._stub([])
        index = person_index.build_person_index(
            [("staff", [path])], ["Williams, Donell", "Nobody, Atall"],
            build_manifest([path]), warn=self.warnings.append)
        unselected = index["sources"][0]["unselected"]
        self.assertEqual(unselected["Williams, Donell"]["pool_total"], 2)
        self.assertEqual(sorted(unselected["Williams, Donell"]["pool"]),
                         ["WILLIAMS, JASON", "WILLIAMS, MARK"])
        # A surname nothing shares is a stronger finding, and it reads differently in the block.
        self.assertEqual(unselected["Nobody, Atall"]["pool_total"], 0)

    def test_the_staff_pass_is_cached_so_two_nodes_pay_once(self) -> None:
        path = self._staff_file()
        self._stub([("Oyler, Ken", "OYLER, KENNETH")])
        args = ([("staff", [path])], ["Oyler, Ken"], build_manifest([path]))
        first = person_index.build_person_index(*args, warn=self.warnings.append)
        after_first = self.calls
        second = person_index.build_person_index(*args, warn=self.warnings.append)
        self.assertEqual(self.calls, after_first)          # no second model call
        self.assertEqual(second["sources"][0]["selected"], first["sources"][0]["selected"])

    def test_the_kill_switch_makes_no_model_call(self) -> None:
        path = self._staff_file()
        self._stub([("Oyler, Ken", "OYLER, KENNETH")])
        os.environ["OTL_PERSON_INDEX"] = "0"
        index = person_index.build_person_index(
            [("staff", [path])], ["Oyler, Ken"], build_manifest([path]), warn=self.warnings.append)
        self.assertFalse(index["enabled"])
        self.assertEqual(index["sources"], [])
        self.assertEqual(self.calls, 0)
        self.assertEqual(person_index.person_index_block(index, ["Oyler, Ken"]), "")
        self.assertEqual(person_index.index_json(index), {})

    def test_an_unreadable_workbook_is_listed_and_the_node_still_runs(self) -> None:
        broken = self.dir / "SEPCO 2022 Historical POB report.xlsx"
        broken.write_text("this is not a workbook", encoding="utf-8")
        self._stub([])
        index = person_index.build_person_index(
            [("pob", [str(broken)])], ["Oyler, Ken"], build_manifest([str(broken)]),
            warn=self.warnings.append)
        self.assertEqual([u["name"] for u in index["unreadable"]],
                         ["SEPCO 2022 Historical POB report.xlsx"])
        self.assertTrue(index["unreadable"][0]["errors"])
        block = person_index.person_index_block(index, ["Oyler, Ken"])
        self.assertIn("UNREADABLE", block)
        self.assertIn("Do not treat it as missing", block)

    def test_no_people_means_an_empty_index(self) -> None:
        path = self._staff_file()
        self._stub([("Oyler, Ken", "OYLER, KENNETH")])
        index = person_index.build_person_index([("staff", [path])], [], build_manifest([path]),
                                                warn=self.warnings.append)
        self.assertEqual(index["sources"], [])
        self.assertEqual(self.calls, 0)


class BlockTests(unittest.TestCase):
    """The bounded prompt block. Every bound has to state what it dropped."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self._real = person_index.llm.structured
        person_index._INDEX_CACHE.clear()

    def tearDown(self) -> None:
        person_index.llm.structured = self._real
        person_index._INDEX_CACHE.clear()
        self.tmp.cleanup()

    def _index(self, sheets: dict[str, list[list]], people: list[str],
               pairs: list[tuple[str, str]], name: str = "staff.xlsx") -> dict:
        path = str(_write_workbook(self.dir / name, sheets))

        def _structured(node, system, user, schema, **kwargs):
            head = user.split("These are the people under dispute")[0]
            chunk = [line[2:] for line in head.splitlines() if line.startswith("- ")]
            return schema(matches=[{"person": p, "name_in_file": v, "confidence": "high",
                                    "why": "surname and given name match"}
                                   for p, v in pairs if v in chunk])

        person_index.llm.structured = _structured
        return person_index.build_person_index([("staff", [path])], people, build_manifest([path]),
                                               warn=lambda m: None)

    def test_the_block_names_the_subject_the_roles_and_the_roles_absent(self) -> None:
        index = self._index({"Staff Listing": _staff_rows(_STAFF_WORKER, ["OYLER, KEN"])},
                            ["Oyler, Ken"], [("Oyler, Ken", "OYLER, KEN")])
        block = person_index.person_index_block(index, ["Oyler, Ken"])
        self.assertIn("subject: Worker", block)
        self.assertIn("reports_to=Supervisory Organization - ID", block)
        self.assertIn("not present:", block)
        self.assertIn("job_profile", block.split("not present:")[1])
        self.assertIn("Oyler, Ken → OYLER, KEN", block)
        self.assertIn("person_index()", block)
        self.assertIn("trust your read", block)          # the PRIOR-not-a-guarantee doctrine

    def test_a_person_with_no_selection_reads_as_a_finding(self) -> None:
        index = self._index(
            {"Staff Listing": _staff_rows(_STAFF_REPORTING, ["WILLIAMS, JASON", "WILLIAMS, MARK"])},
            ["Williams, Donell", "Nobody, Atall"], [])
        block = person_index.person_index_block(index, ["Williams, Donell", "Nobody, Atall"])
        self.assertIn("Williams, Donell → no value selected. 2 value(s) share this surname", block)
        self.assertIn("Nobody, Atall → no value selected, and NO value in this file shares", block)
        self.assertIn("Absence is a finding", block)

    def test_tabs_sharing_a_shape_collapse_to_one_line(self) -> None:
        # The 2021 workbook has 18 name-bearing tabs. Listing each would be most of the block.
        sheets = {f"IG{i}": _staff_rows(_STAFF_WORKER, ["OYLER, KEN"], key_start=100 * i)
                  for i in range(1, 10)}
        index = self._index(sheets, ["Oyler, Ken"], [("Oyler, Ken", "OYLER, KEN")])
        block = person_index.person_index_block(index, ["Oyler, Ken"])
        self.assertIn("more tabs, same columns", block)
        self.assertEqual(block.count("subject: Worker"), 1)
        self.assertIn("9 name-bearing tab(s)", block)

    def test_the_block_shrinks_as_the_person_count_grows(self) -> None:
        self.assertEqual(person_index._names_shown(["a"] * 5), 4)
        self.assertEqual(person_index._names_shown(["a"] * 20), 2)
        self.assertEqual(person_index._names_shown(["a"] * 50), 1)

    def test_extra_candidate_spellings_are_counted_not_dropped_silently(self) -> None:
        names = [f"OYLER, K{i}" for i in range(6)]
        index = self._index({"Staff Listing": _staff_rows(_STAFF_REPORTING, names)},
                            ["Oyler, Ken"], [("Oyler, Ken", n) for n in names])
        real = person_index._names_shown
        person_index._names_shown = lambda people: 2
        try:
            block = person_index.person_index_block(index, ["Oyler, Ken"])
        finally:
            person_index._names_shown = real
        self.assertIn("more candidate spelling(s) for Oyler, Ken", block)

    def test_the_whole_block_is_capped_and_says_it_was_capped(self) -> None:
        index = self._index({"Staff Listing": _staff_rows(_STAFF_REPORTING, ["OYLER, KEN"])},
                            ["Oyler, Ken"], [("Oyler, Ken", "OYLER, KEN")])
        real = person_index._MAX_BLOCK_CHARS
        person_index._MAX_BLOCK_CHARS = 400
        try:
            block = person_index.person_index_block(index, ["Oyler, Ken"])
        finally:
            person_index._MAX_BLOCK_CHARS = real
        self.assertIn("truncated at 400 characters", block)


class StagedIndexTests(unittest.TestCase):
    """The zero-token channel: `index_json` on disk, read back by `person_index()` in the sandbox."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self._real = person_index.llm.structured
        person_index._INDEX_CACHE.clear()

    def tearDown(self) -> None:
        person_index.llm.structured = self._real
        person_index._INDEX_CACHE.clear()
        self.tmp.cleanup()

    def _index(self) -> dict:
        path = str(_write_workbook(self.dir / "staff.xlsx", {
            "Staff Listing": _staff_rows(_STAFF_REPORTING, ["OYLER, KENNETH", "PARK, SUSAN"])}))

        def _structured(node, system, user, schema, **kwargs):
            return schema(matches=[{"person": "Oyler, Ken", "name_in_file": "OYLER, KENNETH",
                                    "confidence": "high", "why": ""}])

        person_index.llm.structured = _structured
        return person_index.build_person_index([("staff", [path])], ["Oyler, Ken"],
                                               build_manifest([path]), warn=lambda m: None)

    def test_index_json_drops_the_universe_and_keeps_what_the_agent_reasons_over(self) -> None:
        payload = person_index.index_json(self._index())
        source = payload["sources"][0]
        self.assertNotIn("values", source)
        entry = source["entries"][0]
        self.assertNotIn("values", entry)
        self.assertEqual(entry["n_values"], 2)
        self.assertIn("columns", entry)
        self.assertIn("OYLER, KENNETH", entry["row_data"])
        # It must survive a JSON round trip, because that is how it reaches the sandbox.
        self.assertEqual(json.loads(json.dumps(payload, default=str))["sources"][0]["role"], "staff")

    def test_the_sandbox_reads_the_staged_index(self) -> None:
        payload = person_index.index_json(self._index())
        ex = CodeExecutor("test-person-index", "location", read_roots=[str(self.dir)],
                          scratch_base=self.dir / "scratch", person_index=payload)
        code = ("idx = person_index()\n"
                "one = person_index(person='Oyler, Ken', role='staff')\n"
                "save_result({'roles': [s['role'] for s in idx['sources']],\n"
                "             'name': one['sources'][0]['selected']['Oyler, Ken'][0]['name_in_file'],\n"
                "             'reports_to': list(one['sources'][0]['entries'][0]['columns']"
                "['reports_to']),\n"
                "             'rows': list(one['sources'][0]['entries'][0]['row_data'])})\n")
        got = ex.run(code, inputs={}, timeout=60)
        self.assertTrue(got["ok"], got)
        self.assertEqual(got["result"]["roles"], ["staff"])
        self.assertEqual(got["result"]["name"], "OYLER, KENNETH")
        self.assertEqual(got["result"]["reports_to"], ["Manager ID"])
        self.assertEqual(got["result"]["rows"], ["OYLER, KENNETH"])

    def test_a_node_passing_no_index_leaves_no_stale_file(self) -> None:
        scratch = self.dir / "scratch"
        staged = CodeExecutor("test-person-index", "job_title", read_roots=[str(self.dir)],
                              scratch_base=scratch, person_index=person_index.index_json(self._index()))
        staged.run("pass", inputs={}, timeout=60)
        self.assertTrue((staged.scratch / "person_index.json").exists())

        bare = CodeExecutor("test-person-index", "job_title", read_roots=[str(self.dir)],
                            scratch_base=scratch)
        got = bare.run("save_result({'idx': person_index()})", inputs={}, timeout=60)
        self.assertFalse((bare.scratch / "person_index.json").exists())
        self.assertTrue(got["ok"], got)
        self.assertEqual(got["result"]["idx"], {})


if __name__ == "__main__":
    unittest.main()
