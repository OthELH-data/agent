from __future__ import annotations

import json
import os
import unittest

from otl.matching_evidence import build_shards, mapping_rows, prepare_packets


@unittest.skipUnless(os.getenv("RUN_REAL_MATCHING_TESTS") == "1", "set RUN_REAL_MATCHING_TESTS=1")
class RealMatchingCases(unittest.TestCase):
    def test_great_white_50_person_cfr_packets(self):
        result_path = "orchestration-flow/.otl_scratch/GreatWhite_2023_0004/job_title/result.json"
        mapping = ("/Users/harshakolachina/Documents/Projects/Shell JV Audit/Data Files/OTL Information/"
                   "staff_listing_with_CFR_mapping 1.xlsx")
        if not (os.path.exists(result_path) and os.path.exists(mapping)):
            self.skipTest("real job-title result or CFR mapping unavailable")
        result = json.load(open(result_path, encoding="utf-8"))
        positions = result.get("identified_names_staff_listing") or {}
        state = {"job_title": {
            "positions_by_person": {person: [position] for person, position in positions.items()},
            "selected_job_profiles_by_person": {},
            "identified_job_data": result.get("identified_job_data") or {},
            "identified_job_descriptions": result.get("identified_job_descriptions") or {},
        }}
        rows = mapping_rows(mapping)
        packets = prepare_packets(state, mapping)
        shards, oversize = build_shards(packets)
        self.assertEqual(len(rows), 4698)
        self.assertEqual(len(packets), 50)
        self.assertLessEqual(len(shards), 2)
        self.assertFalse(oversize)
        self.assertTrue(all(5 <= len(packet.candidates) <= 8 for packet in packets))


if __name__ == "__main__":
    unittest.main()
