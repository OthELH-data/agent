from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from otl.job_title_evidence import build_shards, prepare_evidence
from otl.person_index import _NameSelection, build_person_index
from otl.workbook_manifest import build_manifest


@unittest.skipUnless(os.getenv("RUN_REAL_JOB_TITLE_TESTS") == "1", "set RUN_REAL_JOB_TITLE_TESTS=1")
class RealJobTitleCases(unittest.TestCase):
    def test_great_white_50_person_structured_packet(self):
        inputs_path = "orchestration-flow/.otl_scratch/GreatWhite_2023_0004/job_title/inputs.json"
        if not os.path.exists(inputs_path):
            self.skipTest("saved real-case inputs unavailable")
        inputs = json.load(open(inputs_path, encoding="utf-8"))
        staff, catalogue = inputs["staff_listing"][0], inputs["jobs_catalogue"][0]
        if not (os.path.exists(staff) and os.path.exists(catalogue)):
            self.skipTest("real staff/catalogue sources unavailable")
        people = inputs["disputed_people"]
        manifest = build_manifest([staff, catalogue], people=people)
        with patch("otl.person_index.llm.structured", return_value=_NameSelection(matches=[])) as selector:
            index = build_person_index([("staff", [staff])], people, manifest)
        prepared = prepare_evidence(
            state={"case": {"year": "2023", "asset": "Great White"},
                   "jade_analyzer": {"ledger_rows": []}},
            index=index, people=people, staff_listing=staff, jobs_catalogue=catalogue,
        )
        shards, oversize = build_shards(prepared.packets)
        self.assertEqual(len(people), 50)
        self.assertGreaterEqual(sum(bool(packet.staff_title_periods) for packet in prepared.packets), 49)
        self.assertLessEqual(len(shards), 2)
        self.assertFalse(oversize)
        self.assertLessEqual(selector.call_count, 1)


if __name__ == "__main__":
    unittest.main()
