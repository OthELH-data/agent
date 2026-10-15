"""Tests for the major-construction (MCOH) screen jade_analyzer runs per person.

Everything here runs offline. No workbook is opened and no LLM is called: the tests feed a finish payload
straight into `_to_jade_state`, and a JadeState straight into the orchestrator digest and the drafter's
presentation layer.

The behaviours worth guarding:

* THE FLAG CANNOT CONTRADICT THE READING. `major_construction_people` and `major_construction_charge` are
  derived IN CODE from `major_construction_by_person`, so a model that reports a flag it did not read off
  the cells gets ignored.
* The orchestrator always sees the flag, and sees the people and the ledger values behind it only when
  the screen fired. A screen that fired for one person of five is not a whole-case basis.
* The drafter is handed the MCOH facts only when the screen fired, and never the raw cell values.
* The authority sentence quotes the settlement's own OTL definition (Attachment 1 §1). No separate
  major-construction provision is asserted, and the sentence stands on its own when `matching` never ran.
"""
from __future__ import annotations

import unittest

from otl.agents.drafter import _authority_application, _nl_summary_digest
from otl.agents.jade_analyzer import _to_jade_state
from otl.agents.orchestrator import _salient_nl_summaries
from otl.authority import OTL_MAJOR_CONSTRUCTION_PROVISION
from otl.state import JadeState

# The fields `jade_analyzer_node` computes in code before the loop runs and passes into the mapper. The
# model cannot write them, so every call to `_to_jade_state` has to supply them.
_HARVEST = {
    "ledger_tabs_read": ["JADE"],
    "ledger_person_values": ["ANDERSON, KIRK", "Not assigned"],
    "people_found_in_ledger": ["Kirk Anderson", "Dana Reyes"],
}


def _mapped(payload: dict) -> JadeState:
    return _to_jade_state(payload, 100_000.0, dict(_HARVEST))


class ScreenMappingTests(unittest.TestCase):
    def test_the_summary_fields_are_derived_from_the_per_person_map(self) -> None:
        sub = _mapped({
            "major_construction_by_person": {"Kirk Anderson": True, "Dana Reyes": False},
            "jv_object_type_by_person": {"Kirk Anderson": "C142", "Dana Reyes": "O999"},
            "major_construction_basis": "read 'WBS:JV Object Type'; C142 on Anderson's rows.",
        })

        self.assertTrue(sub.major_construction_charge)
        self.assertEqual(sub.major_construction_people, ["Kirk Anderson"])
        self.assertEqual(sub.major_construction_by_person,
                         {"Kirk Anderson": True, "Dana Reyes": False})
        self.assertEqual(sub.jv_object_type_by_person["Dana Reyes"], "O999")
        self.assertIn("WBS:JV Object Type", sub.major_construction_basis)
        self.assertEqual(sub.people_named_count, len(sub.people_found_in_ledger))

    def test_people_named_count_is_derived_from_the_grounded_list(self) -> None:
        sub = _to_jade_state(
            {"people_named_count": 999},
            100_000.0,
            {"people_found_in_ledger": ["One Person", "Two Person", "Three Person"]},
        )

        self.assertEqual(sub.people_named_count, 3)

    def test_the_model_cannot_assert_a_flag_it_did_not_read_off_the_cells(self) -> None:
        # The payload claims the charge outright and names a person, but its own per-person reading is
        # empty. The reading wins: these two fields are derived, never read off the payload.
        sub = _mapped({
            "major_construction_charge": True,
            "major_construction_people": ["Kirk Anderson"],
            "major_construction_by_person": {},
        })

        self.assertFalse(sub.major_construction_charge)
        self.assertEqual(sub.major_construction_people, [])

    def test_the_model_cannot_deny_a_flag_its_own_reading_supports(self) -> None:
        sub = _mapped({
            "major_construction_charge": False,
            "major_construction_by_person": {"Kirk Anderson": True},
        })

        self.assertTrue(sub.major_construction_charge)
        self.assertEqual(sub.major_construction_people, ["Kirk Anderson"])

    def test_a_ledger_without_the_column_leaves_the_maps_empty(self) -> None:
        sub = _mapped({"major_construction_basis": "this ledger carries no WBS:JV Object Type column."})

        self.assertFalse(sub.major_construction_charge)
        self.assertEqual(sub.major_construction_by_person, {})
        self.assertEqual(sub.jv_object_type_by_person, {})
        self.assertEqual(sub.major_construction_people, [])

    def test_a_payload_that_is_not_a_map_is_tolerated(self) -> None:
        # The loop's payload is free-form, so a model may hand back a list or a string here. It must not
        # raise, and it must not be read as a flag.
        for junk in (["Kirk Anderson"], "Kirk Anderson", None, 1):
            with self.subTest(junk=junk):
                sub = _mapped({"major_construction_by_person": junk,
                               "jv_object_type_by_person": junk})
                self.assertFalse(sub.major_construction_charge)
                self.assertEqual(sub.major_construction_by_person, {})
                self.assertEqual(sub.jv_object_type_by_person, {})

    def test_maps_are_coerced_and_restricted_to_grounded_ledger_people(self) -> None:
        sub = _mapped({
            "major_construction_by_person": {
                "Kirk Anderson": "yes", "Dana Reyes": 0, "Invented Person": True, 7: 0,
            },
            "jv_object_type_by_person": {
                "Kirk Anderson": 42, "Dana Reyes": "O999", "Invented Person": "C142",
            },
        })

        self.assertEqual(
            sub.major_construction_by_person, {"Kirk Anderson": True, "Dana Reyes": False}
        )
        self.assertEqual(
            sub.jv_object_type_by_person, {"Kirk Anderson": "42", "Dana Reyes": "O999"}
        )


class OrchestratorSurfaceTests(unittest.TestCase):
    """What the supervisor gets to weigh. Nothing in code routes on the flag — this IS the whole path."""

    def _findings(self, sub: JadeState) -> dict:
        return _salient_nl_summaries({"jade_analyzer": sub})["jade_analyzer"]

    def test_the_flag_and_the_reading_behind_it_reach_the_supervisor(self) -> None:
        jade = self._findings(_mapped({
            "major_construction_by_person": {"Kirk Anderson": True, "Dana Reyes": False},
            "jv_object_type_by_person": {"Kirk Anderson": "C142", "Dana Reyes": "O999"},
            "major_construction_basis": "read 'JV Object Type' code per row.",
        }))

        self.assertTrue(jade["major_construction_charge"])
        # WHICH people it covers has to travel with the flag: the supervisor scopes its finalize to them.
        self.assertEqual(jade["major_construction_people"], ["Kirk Anderson"])
        self.assertEqual(jade["jv_object_type_by_person"]["Kirk Anderson"], "C142")
        self.assertIn("JV Object Type", jade["major_construction_basis"])

    def test_a_screen_that_did_not_fire_carries_its_basis_but_no_people(self) -> None:
        jade = self._findings(_mapped({
            "major_construction_by_person": {"Dana Reyes": False},
            "jv_object_type_by_person": {"Dana Reyes": "O999"},
            "major_construction_basis": "read 'WBS:JV Object Type'; no registered code on any row.",
        }))

        self.assertFalse(jade["major_construction_charge"])
        self.assertIn("no registered code", jade["major_construction_basis"])
        self.assertNotIn("major_construction_people", jade)
        self.assertNotIn("jv_object_type_by_person", jade)

    def test_the_flag_is_present_even_when_the_screen_could_not_run(self) -> None:
        # Absent keys would read as "the screen was never mentioned". False says it ran and found nothing.
        jade = self._findings(_mapped({}))

        self.assertIn("major_construction_charge", jade)
        self.assertFalse(jade["major_construction_charge"])
        self.assertNotIn("major_construction_basis", jade)


class DrafterPresentationTests(unittest.TestCase):
    def test_the_screen_reaches_the_response_writer_only_when_it_fired(self) -> None:
        fired = _nl_summary_digest({"jade_analyzer": _mapped({
            "major_construction_by_person": {"Kirk Anderson": True},
            "jv_object_type_by_person": {"Kirk Anderson": "C142"},
            "major_construction_basis": "read 'WBS:JV Object Type'.",
        })})["jade_analyzer"]

        self.assertTrue(fired["major_construction_charge"])
        self.assertEqual(fired["major_construction_people"], ["Kirk Anderson"])
        self.assertIn("major_construction_basis", fired)
        # Raw ledger cell values are trace material and never travel to the client-facing prose.
        self.assertNotIn("jv_object_type_by_person", fired)

    def test_a_negative_screen_is_kept_out_of_the_response_writer(self) -> None:
        # A False flag, or a basis line saying the ledger has no such column, is a fact AGAINST the
        # operator's charge. The supervisor weighs it; the drafter has no use for it.
        quiet = _nl_summary_digest({"jade_analyzer": _mapped({
            "major_construction_by_person": {"Dana Reyes": False},
            "major_construction_basis": "read 'WBS:JV Object Type'; no registered code on any row.",
        })})["jade_analyzer"]

        self.assertNotIn("major_construction_charge", quiet)
        self.assertNotIn("major_construction_people", quiet)
        self.assertNotIn("major_construction_basis", quiet)


class AuthorityQuotationTests(unittest.TestCase):
    """The billing standard quoted for an MCOH charge is the COUA's own Major Construction clause."""

    _STATE = {
        "case": {"case_id": "GreatWhite_2023_0003"},
        "jade_analyzer": None,  # replaced per test
    }

    def _state(self, **extra) -> dict:
        sub = _mapped({"major_construction_by_person": {"Kirk Anderson": True},
                       "jv_object_type_by_person": {"Kirk Anderson": "C142"}})
        return {"case": {"case_id": "GreatWhite_2023_0003"}, "jade_analyzer": sub, **extra}

    def test_the_sentence_stands_alone_when_matching_never_ran(self) -> None:
        # An MCOH case can be finalized straight off the ledger, so this is the only authority such a
        # case carries. Before the screen existed, no `matching` meant no authority sentence at all.
        text = _authority_application(self._state())

        self.assertIn("major construction overhead", text)
        self.assertIn(str(OTL_MAJOR_CONSTRUCTION_PROVISION["section"]), text)
        self.assertIn(str(OTL_MAJOR_CONSTRUCTION_PROVISION["instrument"]), text)
        # The abbreviation is a ledger classification code and stays out of client-facing prose.
        self.assertNotIn("MCOH", text)

    def test_no_generic_offsite_provision_is_substituted(self) -> None:
        text = _authority_application(self._state())

        # The proposition comes out of the COUA registry entry's own statement, not out of prose
        # written here, and not out of the generic Attachment 1 §1 offsite-labor stand-in this used to
        # quote before the COUA clause was available.
        statement = str(OTL_MAJOR_CONSTRUCTION_PROVISION["statement"]).rstrip(".")
        self.assertIn(statement[0].lower() + statement[1:], text)
        self.assertNotIn("Attachment 1 §1", text)
        self.assertNotIn("Attachment 1 §2", text)
        self.assertNotIn("Attachment 1 §3", text)

    def test_a_negative_screen_produces_no_sentence(self) -> None:
        state = {"case": {"case_id": "GreatWhite_2023_0003"},
                 "jade_analyzer": _mapped({"major_construction_by_person": {"Dana Reyes": False}})}

        self.assertEqual(_authority_application(state), "")

    def test_each_instrument_is_named_once_when_a_category_route_also_applies(self) -> None:
        # The MCOH sentence quotes the COUA; the technical/category sentences quote the GOM Settlement
        # Agreement. Two different instruments, so "Under the" appears once per instrument — and "The
        # same agreement's" only ties the technical and category sentences together, never the MCOH one.
        state = self._state(matching={"identified_cfr_category": "Regulatory Activity"},
                            job_title={"technical_job_title": True})
        text = _authority_application(state)

        self.assertEqual(text.count("Under the"), 1)
        self.assertEqual(text.count("The same agreement's"), 0)
        self.assertIn("major construction overhead", text)
        self.assertNotIn("GOM Settlement Agreement", text)

    def test_an_out_of_scope_asset_yields_no_authority_at_all(self) -> None:
        state = self._state()
        state["case"] = {"case_id": "Elsewhere_2023_0001", "asset": "Some Other Platform"}

        self.assertEqual(_authority_application(state), "")


if __name__ == "__main__":
    unittest.main()
