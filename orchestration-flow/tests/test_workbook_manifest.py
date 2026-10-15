"""Tests for the deterministic workbook manifest.

Most of these build a small workbook in a temp directory, so they run anywhere. The few that need the
real corpus are guarded on `Data Files` existing and skip cleanly when it does not.

The two behaviours worth guarding hardest:

* A merged banner row must NOT be picked as the header. All five agents inherit the manifest, so a
  confidently wrong header row is worse than no manifest at all.
* The header row index must be part of the shape fingerprint. The same 22-column labor header sits at
  row 0 on 50 tabs and row 1 on 5 more in one corpus workbook. Collapsing those two groups would tell
  an agent to start reading a row too early on five people.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from otl._bootstrap import DATA_ROOT    # importing otl first is what puts `extractor` on sys.path
from otl.code_exec import CodeExecutor
from otl.workbook_manifest import build_manifest, manifest_block, resolve_person_tabs

from extractor.tab_schemas import choose_header_row  # noqa: E402 — needs the otl bootstrap above

# The 63-tab exhibit from exception 0004 — the widest shape variety in the corpus.
BIG_WORKBOOK = (DATA_ROOT / "Exception Reports and Rebuttals by year/2023/Great White/Rebuttal 0"
                / "Audit Report/0004/Affiliate Labor Exception Support.xlsx")

_LABOR_HEADER = ["Business", "Month", "Employee ID", "Employee Reporting Name", "Position"]
_HELI_HEADER = ["ASSET ID", "ASSET", "CALL SIGN", "TRIP ID", "LEG", "NAME", "COMPANY"]


def _write_workbook(path: Path, sheets: dict[str, list[list]]) -> Path:
    """Write a real .xlsx so the manifest opens it the same way it opens a corpus file."""
    from openpyxl import Workbook

    wb = Workbook()
    wb.remove(wb.active)
    for title, rows in sheets.items():
        ws = wb.create_sheet(title=title[:31])
        for row in rows:
            ws.append(row)
    wb.save(path)
    return path


def _labor_tab(header_row: int) -> list[list]:
    """A labor-shaped tab with `header_row` blank rows above the header."""
    rows: list[list] = [[] for _ in range(header_row)]
    rows.append(list(_LABOR_HEADER))
    rows.append(["Shell", "2023-01", "E100", "Someone", "Technician"])
    return rows


class HeaderRowDetectionTests(unittest.TestCase):
    """`choose_header_row` — pure, no file needed."""

    def test_banner_row_is_not_chosen_as_the_header(self) -> None:
        # A merged banner repeats ONE value across its cells. The real header is the row below.
        banner = ["Personnel On Board"] * 15
        header = ["Region", "Area", "Location", "Location Code", "Current POB", "Max POB",
                  "Location Type", "Helipass Id", "First Name", "Last Name", "Project", "Company",
                  "Position", "Arrival Date", "Departure Date", "Essential", "Notes"]
        data = ["GOM", "Houma", "Great White", "GW01", 40, 60, "Platform", "H1", "Ken", "Oyler",
                "P1", "Shell", "Tech", "2023-01-01", "2023-01-14", "Y", ""]

        idx, confidence = choose_header_row([banner, header, data])

        self.assertEqual(idx, 1, "the banner row was picked instead of the header")
        self.assertEqual(confidence, "label-row")

    def test_headerless_sheet_is_reported_as_positional(self) -> None:
        rows = [[1, 2.5, 3, 4], [5, 6.5, 7, 8], [9, 10.5, 11, 12]]

        idx, confidence = choose_header_row(rows)

        self.assertIsNone(idx)
        self.assertEqual(confidence, "no-header(positional)")

    def test_empty_sheet_is_reported_as_empty(self) -> None:
        idx, confidence = choose_header_row([[], [None, None], []])

        self.assertIsNone(idx)
        self.assertEqual(confidence, "empty")


class PersonResolutionTests(unittest.TestCase):
    """`resolve_person_tabs` — ambiguity is reported, never guessed."""

    def test_two_tabs_share_a_surname_and_are_both_reported(self) -> None:
        found = resolve_person_tabs(["Adams N", "Adams S", "Oyler"], ["Adams"])

        self.assertEqual(found["Adams"]["tabs"], ["Adams N", "Adams S"])
        self.assertTrue(found["Adams"]["ambiguous"])

    def test_given_name_confirms_a_single_tab(self) -> None:
        found = resolve_person_tabs(["Oyler", "Reasor"], ["Ken Oyler", "Oyler, Ken"])

        self.assertEqual(found["Ken Oyler"]["tabs"], ["Oyler"])
        self.assertFalse(found["Ken Oyler"]["ambiguous"])
        # "Surname, Given" and "Given Surname" have to resolve identically.
        self.assertEqual(found["Oyler, Ken"]["tabs"], ["Oyler"])

    def test_an_absent_person_is_recorded_with_no_tabs(self) -> None:
        # Recorded rather than dropped, so the agent is told the tab does not exist instead of looking.
        found = resolve_person_tabs(["Oyler", "Reasor"], ["Zzzzz"])

        self.assertEqual(found["Zzzzz"]["tabs"], [])
        self.assertEqual(found["Zzzzz"]["match"], "none")
        self.assertFalse(found["Zzzzz"]["ambiguous"])


class SynthesisHintTests(unittest.TestCase):
    """The extractor already named these people. `resolve_person_tabs` reads its notes instead of
    re-deriving the mapping, and `person_tab_hints` is what builds those notes out of the synthesis."""

    # Close to the real thing: the notes for `Adams N` name Nancy outright, which is the fact that tells
    # `Adams N` from `Adams S`. A tab name alone cannot.
    HINTS = {
        "Adams N": "Adams N Labor detail for a single affiliate employee, Nancy Adams "
                   "(Employee ID 10203215), charged to the joint account.",
        "Adams S": "Adams S Labor detail for Steve Adams (Employee ID 10204471).",
        "Summary": "Summary Roll-up of all affiliate labor by month. Reviewed by Nancy's manager.",
    }

    def test_the_notes_break_a_surname_tie_a_tab_name_cannot(self) -> None:
        found = resolve_person_tabs(["Adams N", "Adams S", "Summary"], ["Nancy Adams"],
                                    tab_hints=self.HINTS)

        hit = found["Nancy Adams"]
        self.assertEqual(hit["tabs"], ["Adams N"])
        self.assertEqual(hit["match"], "surname+synthesis")
        self.assertFalse(hit["ambiguous"])

    def test_the_two_narrowings_compose_when_the_notes_name_two_people(self) -> None:
        # Straight from the corpus: `Adams N` is Nancy's tab, and its notes ALSO discuss Sheila. So the
        # notes leave both Adams tabs standing for Sheila, and the `S` in the tab name is what decides.
        hints = dict(self.HINTS)
        hints["Adams N"] += (" It contrasts her charges with Sheila Adams's, whose rows are on the "
                            "next tab.")

        found = resolve_person_tabs(["Adams N", "Adams S"], ["Sheila Adams", "Nancy Adams"],
                                    tab_hints=hints)

        self.assertEqual(found["Sheila Adams"]["tabs"], ["Adams S"])
        self.assertEqual(found["Sheila Adams"]["match"], "surname+synthesis+given")
        self.assertFalse(found["Sheila Adams"]["ambiguous"])
        # Nancy is named on one tab only, so the notes settle her without the second narrowing.
        self.assertEqual(found["Nancy Adams"]["match"], "surname+synthesis")

    def test_a_person_with_no_named_tab_is_placed_by_the_notes(self) -> None:
        # The affiliate / contractor case: no tab carries their surname, but a shared tab's notes do.
        hints = {"Summary": "Summary Roll-up by month. Includes contractor Bob Kihm's hours."}

        found = resolve_person_tabs(["Adams N", "Summary"], ["Bob Kihm"], tab_hints=hints)

        hit = found["Bob Kihm"]
        self.assertEqual(hit["tabs"], ["Summary"])
        self.assertEqual(hit["match"], "synthesis")
        self.assertFalse(hit["own_tab"])

    def test_a_bare_surname_in_prose_is_not_treated_as_a_match(self) -> None:
        # One tab's notes mention several people. A surname on its own, with no given name beside it, is
        # not evidence the tab is about that person.
        hints = {"Summary": "Summary Roll-up by month, approved by Adams and reviewed by Reasor."}

        found = resolve_person_tabs(["Summary"], ["Nancy Adams"], tab_hints=hints)

        self.assertEqual(found["Nancy Adams"]["tabs"], [])
        self.assertEqual(found["Nancy Adams"]["match"], "none")

    def test_the_notes_are_read_in_either_name_order_and_through_a_middle_name(self) -> None:
        cases = {
            "Adams, Nancy": "Adams N Detail for Adams, Nancy (ID 10203215).",
            "Nancy J Adams": "Adams N Detail for Nancy J Adams (ID 10203215).",
            "N. Adams": "Adams N Detail for N Adams (ID 10203215).",
            "Nancy Adams": "Adams N Detail for Nancy Adams (ID 10203215).",
        }
        for person, note in cases.items():
            with self.subTest(person=person):
                found = resolve_person_tabs(["Adams N", "Adams S"], [person],
                                            tab_hints={"Adams N": note})
                self.assertEqual(found[person]["tabs"], ["Adams N"])
                self.assertEqual(found[person]["match"], "surname+synthesis")

    def test_a_tab_the_manifest_cannot_use_is_never_matched_from_notes(self) -> None:
        # Hints for a tab that is not in the usable sheet list are ignored, so a person can never be
        # pointed at a tab the response filter dropped.
        found = resolve_person_tabs(["Adams N"], ["Bob Kihm"],
                                    tab_hints={"Operator Response": "Kihm, Bob rebuttal"})

        self.assertEqual(found["Bob Kihm"]["tabs"], [])

    def test_hints_reach_resolution_through_build_manifest_by_file_name(self) -> None:
        # The synthesis records a file NAME; the agents hold an absolute path. build_manifest bridges it.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = _write_workbook(Path(tmp.name) / "Affiliate Labor.xlsx", {
            "Adams N": _labor_tab(0),
            "Adams S": _labor_tab(0),
        })

        wb = build_manifest([str(path)], people=["Nancy Adams"],
                            tab_hints={"Affiliate Labor.xlsx": self.HINTS})["workbooks"][0]

        self.assertEqual(wb["person_tabs"]["Nancy Adams"]["tabs"], ["Adams N"])
        self.assertEqual(wb["person_tabs"]["Nancy Adams"]["match"], "surname+synthesis")

    def test_person_tab_hints_builds_the_notes_from_a_synthesis(self) -> None:
        from otl.agents._common import person_tab_hints

        state = {"synthesis": {"attachments": [{
            "file_name": "Affiliate Labor.xlsx",
            "tabs": [
                {"tab_name": "Adams N", "purpose": "a single affiliate employee, Nancy Adams",
                 "notes": "Employee ID 10203215"},
                {"tab_name": "", "purpose": "an unnamed tab that must be skipped"},
            ],
        }]}}

        hints = person_tab_hints(state)

        self.assertIn("Nancy Adams", hints["Affiliate Labor.xlsx"]["Adams N"])
        self.assertIn("10203215", hints["Affiliate Labor.xlsx"]["Adams N"])
        self.assertEqual(list(hints["Affiliate Labor.xlsx"]), ["Adams N"])

    def test_person_tab_hints_is_empty_without_a_synthesis(self) -> None:
        from otl.agents._common import person_tab_hints

        self.assertEqual(person_tab_hints({}), {})
        self.assertEqual(person_tab_hints({"synthesis": {"attachments": []}}), {})

    def test_the_notes_case_renders_as_no_tab_of_their_own(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = _write_workbook(Path(tmp.name) / "shared.xlsx", {"Summary": _labor_tab(0)})
        hints = {"shared.xlsx": {"Summary": "Summary Roll-up including contractor Bob Kihm's hours."}}

        block = manifest_block(build_manifest([str(path)], people=["Bob Kihm"], tab_hints=hints))

        self.assertIn("no tab of their own; named in the notes for Summary", block)
        # They were placed, so the no-tab gating line is not needed.
        self.assertNotIn("NOT out of scope", block)


class ManifestGroupingTests(unittest.TestCase):
    """Grouping and ordering, against a real (small) workbook on disk."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_tabs_with_the_same_shape_collapse_into_one_group(self) -> None:
        path = _write_workbook(self.tmp / "grouped.xlsx", {
            "Adams": _labor_tab(0),
            "Bazan": _labor_tab(0),
            "Oyler": [list(_HELI_HEADER), ["A1", "GW", "N1", "T1", 1, "Ken Oyler", "Shell"]],
        })

        wb = build_manifest([str(path)])["workbooks"][0]

        self.assertEqual(wb["n_sheets"], 3)
        self.assertEqual(len(wb["shapes"]), 2, "three tabs should collapse to two shapes")
        # Biggest group first, and IDs are assigned after that sort.
        self.assertEqual(wb["shapes"][0]["id"], "S1")
        self.assertEqual(wb["shapes"][0]["n_tabs"], 2)
        self.assertEqual(sorted(wb["shapes"][0]["tabs"]), ["Adams", "Bazan"])
        self.assertEqual(wb["tabs"]["Oyler"]["shape"], "S2")

    def test_the_same_header_at_a_different_row_is_a_different_shape(self) -> None:
        # This is the corpus case that makes the header row index part of the fingerprint.
        path = _write_workbook(self.tmp / "rows.xlsx", {
            "Adams": _labor_tab(0),
            "Banga": _labor_tab(1),
        })

        wb = build_manifest([str(path)])["workbooks"][0]

        self.assertEqual(len(wb["shapes"]), 2)
        self.assertEqual(wb["tabs"]["Adams"]["header_row"], 0)
        self.assertEqual(wb["tabs"]["Banga"]["header_row"], 1)

    def test_a_headerless_tab_records_no_header_row(self) -> None:
        path = _write_workbook(self.tmp / "positional.xlsx", {
            "Sheet1": [[1, 2, 3, 4], [5, 6, 7, 8]],
        })

        wb = build_manifest([str(path)])["workbooks"][0]

        self.assertIsNone(wb["tabs"]["Sheet1"]["header_row"])
        self.assertEqual(wb["tabs"]["Sheet1"]["confidence"], "no-header(positional)")

    def test_a_missing_path_is_reported_unreadable_not_dropped(self) -> None:
        manifest = build_manifest([str(self.tmp / "nope.xlsx")])

        self.assertEqual(manifest["workbooks"], [])
        self.assertEqual(len(manifest["unreadable"]), 1)
        self.assertIn("not found", str(manifest["unreadable"][0]["errors"]))

    def test_a_file_no_reader_can_open_is_reported_with_its_errors(self) -> None:
        # A real corpus file behaves this way: it looks like a workbook and every engine refuses it.
        bad = self.tmp / "broken.xlsx"
        bad.write_text("this is not a workbook", encoding="utf-8")

        manifest = build_manifest([str(bad)])
        block = manifest_block(manifest)

        self.assertEqual(manifest["workbooks"], [])
        self.assertEqual(len(manifest["unreadable"]), 1)
        self.assertIn("UNREADABLE: broken.xlsx", block)
        self.assertIn("Do not treat it as missing", block)

    def test_a_directory_is_indexed_by_filename_only(self) -> None:
        (self.tmp / "a.xlsx").write_text("x", encoding="utf-8")
        (self.tmp / "notes.txt").write_text("x", encoding="utf-8")

        manifest = build_manifest([str(self.tmp)])

        self.assertEqual(manifest["workbooks"], [], "indexing a directory must open nothing")
        self.assertEqual(len(manifest["directories"]), 1)
        names = {f["name"] for f in manifest["directories"][0]["files"]}
        self.assertEqual(names, {"a.xlsx", "notes.txt"})

    def test_a_non_workbook_file_is_listed_but_not_parsed(self) -> None:
        pdf = self.tmp / "exception.pdf"
        pdf.write_text("x", encoding="utf-8")

        manifest = build_manifest([str(pdf)])

        self.assertEqual(manifest["workbooks"], [])
        self.assertEqual(manifest["unreadable"], [])
        self.assertEqual(manifest["other_files"][0]["name"], "exception.pdf")


class ManifestBlockTests(unittest.TestCase):
    """The rendered prompt block: bounded, and it says what it left out."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _many_shapes(self, n: int) -> dict:
        sheets = {}
        for i in range(n):
            # A distinct label set per tab, so every tab is its own shape.
            sheets[f"Tab{i}"] = [[f"Col{i}_{j}" for j in range(3 + i % 4)], [1, 2, 3]]
        path = _write_workbook(self.tmp / f"many{n}.xlsx", sheets)
        return build_manifest([str(path)])

    def test_shapes_are_capped_and_the_remainder_is_stated(self) -> None:
        manifest = self._many_shapes(20)

        block = manifest_block(manifest, max_groups=8)

        shape_lines = [ln for ln in block.splitlines() if ln.strip().startswith("W1S")]
        self.assertEqual(len(shape_lines), 8)
        self.assertIn("+12 more shapes", block)
        self.assertIn("covering 12 tab(s) not listed here", block)

    def test_shape_ids_are_qualified_by_workbook(self) -> None:
        first = _write_workbook(self.tmp / "one.xlsx", {"Adams": _labor_tab(0)})
        second = _write_workbook(self.tmp / "two.xlsx", {"Oyler": [list(_HELI_HEADER), [1] * 7]})

        block = manifest_block(build_manifest([str(first), str(second)]))

        # Both workbooks number their own shapes from S1, so the rendered IDs must carry the tag.
        self.assertIn("W1S1", block)
        self.assertIn("W2S1", block)

    def test_a_repeated_header_is_written_as_a_back_reference(self) -> None:
        path = _write_workbook(self.tmp / "repeat.xlsx", {
            "Adams": _labor_tab(0),
            "Bazan": _labor_tab(0),
            "Banga": _labor_tab(1),
        })

        block = manifest_block(build_manifest([str(path)]))

        self.assertIn("columns: same as W1S1", block)
        self.assertEqual(block.count("Employee Reporting Name"), 1,
                         "the same column labels should be printed once, not per shape")

    def test_an_empty_manifest_renders_nothing(self) -> None:
        self.assertEqual(manifest_block({}), "")
        self.assertEqual(manifest_block(build_manifest([])), "")

    def test_the_person_lookup_flags_ambiguity(self) -> None:
        path = _write_workbook(self.tmp / "people.xlsx", {
            "Adams N": _labor_tab(0),
            "Adams S": _labor_tab(0),
        })

        block = manifest_block(build_manifest([str(path)], people=["Adams"]))

        self.assertIn("PERSON TABS", block)
        self.assertIn("AMBIGUOUS, confirm before using", block)

    def test_a_person_with_no_tab_is_said_so_plainly(self) -> None:
        path = _write_workbook(self.tmp / "onetab.xlsx", {"Adams N": _labor_tab(0)})

        block = manifest_block(build_manifest([str(path)], people=["Adams", "Zzzzz"]))

        self.assertIn("Zzzzz → onetab.xlsx: NO tab of their own", block)
        # The gating rule: no tab must not read as out of scope. This is the affiliate / contractor case.
        self.assertIn("often an affiliate or a contractor", block)
        self.assertIn("NOT out of scope", block)
        self.assertIn("Never leave a disputed person out of your result object", block)

    def test_the_gating_line_is_omitted_when_everyone_has_a_tab(self) -> None:
        # It costs tokens, so it is only spent when someone actually has no tab.
        path = _write_workbook(self.tmp / "alltabs.xlsx", {"Adams N": _labor_tab(0)})

        block = manifest_block(build_manifest([str(path)], people=["Adams"]))

        self.assertIn("Adams → alltabs.xlsx: Adams N", block)
        self.assertNotIn("NOT out of scope", block)

    def test_a_person_found_in_another_workbook_does_not_trigger_the_gating_line(self) -> None:
        # "Adams" has no tab in the heli file and a tab in the labor file. That is not a no-tab person.
        labor = _write_workbook(self.tmp / "labor.xlsx", {"Adams N": _labor_tab(0)})
        heli = _write_workbook(self.tmp / "heli.xlsx", {"Manifest": [list(_HELI_HEADER), [1] * 7]})

        block = manifest_block(build_manifest([str(labor), str(heli)], people=["Adams"]))

        self.assertNotIn("NOT out of scope", block)


class ChildProcessTests(unittest.TestCase):
    """The disk channel: the sandbox helpers read a staged file, so they cost no tokens."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.path = _write_workbook(self.tmp / "staged.xlsx", {
            "Adams N": _labor_tab(0),
            "Adams S": _labor_tab(0),
            "Oyler": [list(_HELI_HEADER), ["A1", "GW", "N1", "T1", 1, "Ken Oyler", "Shell"]],
            "Kihm": _labor_tab(2),
        })

    def test_a_pre_resolved_miss_does_not_fall_back_to_a_weaker_search(self) -> None:
        # "Nolan" was asked about and has no tab. The pre-resolution already searched properly, so
        # find_tab must return the empty answer rather than re-guessing on the surname.
        manifest = build_manifest([str(self.path)], people=["Nolan"])
        executor = CodeExecutor("test_run", "jade_analyzer", read_roots=[str(self.tmp)],
                                manifest=manifest)

        result = executor.run('print("NOLAN", find_tab("Nolan"))\n', inputs={})

        self.assertIn("NOLAN []", result.get("stdout", ""))

    def test_the_child_reads_the_staged_manifest(self) -> None:
        manifest = build_manifest([str(self.path)], people=["Ken Oyler", "Adams"])
        executor = CodeExecutor("test_run", "jade_analyzer", read_roots=[str(self.tmp)],
                                manifest=manifest)

        result = executor.run(
            'wb = manifest("staged.xlsx")\n'
            'print("SHEETS", len(wb.get("sheets") or []))\n'
            'print("OYLER", [(h["tab"], h["header_row"], len(h["header"])) '
            'for h in find_tab("Ken Oyler")])\n'
            'print("ADAMS", [(h["tab"], h["ambiguous"]) for h in find_tab("Adams")])\n'
            # Kihm was not in the people list, so this exercises the surname fallback.
            'print("KIHM", [(h["tab"], h["header_row"]) for h in find_tab("Bob Kihm")])\n'
            'print("NOBODY", find_tab("Zzzzz"))\n',
            inputs={})
        out = result.get("stdout", "")

        self.assertIn("SHEETS 4", out)
        self.assertIn("OYLER [('Oyler', 0, 7)]", out)
        self.assertIn("ADAMS [('Adams N', True), ('Adams S', True)]", out)
        self.assertIn("KIHM [('Kihm', 2)]", out)
        self.assertIn("NOBODY []", out)

    def test_the_helpers_degrade_quietly_with_no_manifest(self) -> None:
        executor = CodeExecutor("test_run", "jade_analyzer", read_roots=[str(self.tmp)])

        result = executor.run('print("M", manifest())\nprint("F", find_tab("Oyler"))\n', inputs={})
        out = result.get("stdout", "")

        self.assertIn("M {}", out)
        self.assertIn("F []", out)
        self.assertFalse((executor.scratch / "manifest.json").exists())


@unittest.skipUnless(BIG_WORKBOOK.exists(), f"corpus not present at {DATA_ROOT}")
class CorpusTests(unittest.TestCase):
    """The measurements the design was sized on. These are the numbers that justify the module."""

    def test_the_63_tab_exhibit_collapses_to_eight_shapes(self) -> None:
        wb = build_manifest([str(BIG_WORKBOOK)])["workbooks"][0]

        self.assertEqual(wb["n_sheets"], 63)
        self.assertEqual(len(wb["shapes"]), 8)
        # The header row varies INSIDE one workbook, which is why detection runs per sheet.
        distribution: dict = {}
        for info in wb["tabs"].values():
            distribution[info["header_row"]] = distribution.get(info["header_row"], 0) + 1
        self.assertEqual(distribution, {0: 52, 1: 8, 2: 2, 6: 1})
        # Biggest group first: 50 of the 63 tabs share the labor shape at row 0.
        self.assertEqual(wb["shapes"][0]["id"], "S1")
        self.assertEqual(wb["shapes"][0]["n_tabs"], 50)
        self.assertEqual(wb["shapes"][0]["header_row"], 0)

    def test_oyler_is_a_heli_shaped_tab_not_a_labor_tab(self) -> None:
        # The payoff case. "Oyler" IS a tab, but it carries a 26-column heli header at row 1, and the
        # 0003 location run spent two steps discovering that.
        wb = build_manifest([str(BIG_WORKBOOK)], people=["Ken Oyler"])["workbooks"][0]

        hit = wb["person_tabs"]["Ken Oyler"]
        self.assertEqual(hit["tabs"], ["Oyler"])
        self.assertFalse(hit["ambiguous"])
        shape_id = wb["tabs"]["Oyler"]["shape"]
        shape = next(s for s in wb["shapes"] if s["id"] == shape_id)
        self.assertEqual(shape["header_row"], 1)
        self.assertEqual(shape["n_cols"], 26)
        self.assertEqual(shape["header"][0], "ASSET ID")

    def test_the_prompt_block_stays_small(self) -> None:
        from shared.token_accounting import estimate_message_tokens

        manifest = build_manifest([str(BIG_WORKBOOK)], people=["Ken Oyler", "Adams"])
        block = manifest_block(manifest)
        tokens = estimate_message_tokens([{"role": "user", "content": block}])

        # Measured at 916 tokens (3,192 chars). Listing the 63 sheets one by one is about 6,932
        # tokens, so this is roughly 8x smaller. The bound is a regression guard, not a target.
        self.assertLess(tokens, 1100, f"manifest block grew to {tokens} tokens")

    def test_building_it_is_cheap(self) -> None:
        import time

        # Time a COLD build: the profile cache is keyed on (path, size, mtime), so a warm call is a
        # dict hit and would measure nothing.
        from otl import workbook_manifest

        workbook_manifest._PROFILE_CACHE.clear()
        start = time.perf_counter()
        build_manifest([str(BIG_WORKBOOK)])
        elapsed = time.perf_counter() - start

        # Measured at 0.38s against a 10.8-minute baseline run.
        self.assertLess(elapsed, 5.0, f"manifest build took {elapsed:.2f}s")

    def test_the_profile_cache_is_reused_across_agents(self) -> None:
        # One case is one process and all five agents share it, so only the first node pays.
        from otl import workbook_manifest

        workbook_manifest._PROFILE_CACHE.clear()
        build_manifest([str(BIG_WORKBOOK)])
        self.assertEqual(len(workbook_manifest._PROFILE_CACHE), 1)
        build_manifest([str(BIG_WORKBOOK)])
        self.assertEqual(len(workbook_manifest._PROFILE_CACHE), 1)


if __name__ == "__main__":
    unittest.main()
