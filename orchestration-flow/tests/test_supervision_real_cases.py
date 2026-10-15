from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from otl.person_index import _NameSelection, build_person_index
from otl.supervision_evidence import analyze_supervision
from otl.workbook_manifest import build_manifest


@unittest.skipUnless(os.getenv("RUN_REAL_SUPERVISION_TESTS") == "1",
                     "set RUN_REAL_SUPERVISION_TESTS=1")
class RealSupervisionCases(unittest.TestCase):
    def test_great_white_50_person_hierarchy(self):
        inputs_path = "orchestration-flow/.otl_scratch/GreatWhite_2023_0004/job_title/inputs.json"
        if not os.path.exists(inputs_path):
            self.skipTest("saved real-case inputs unavailable")
        inputs = json.load(open(inputs_path, encoding="utf-8"))
        staff, people = inputs["staff_listing"][0], inputs["disputed_people"]
        if not os.path.exists(staff):
            self.skipTest("real staff listing unavailable")
        manifest = build_manifest([staff], people=people)
        with patch("otl.person_index.llm.structured", return_value=_NameSelection(matches=[])) as selector:
            index = build_person_index([("staff", [staff])], people, manifest)
        result = analyze_supervision(index, people)
        self.assertEqual(len(people), 50)
        self.assertEqual(sum(result.above_first_level.values()), 4)
        self.assertEqual(sum(status == "not_above_first_level"
                             for status in result.supervision_status_by_person.values()), 42)
        self.assertEqual(sum(status == "unresolved"
                             for status in result.supervision_status_by_person.values()), 4)
        self.assertEqual(selector.call_count, 1)


if __name__ == "__main__":
    unittest.main()
