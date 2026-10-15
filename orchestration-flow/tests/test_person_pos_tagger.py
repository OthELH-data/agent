"""Offline tests for the optional local JADE proper-noun candidate generator."""
from __future__ import annotations

import unittest

from otl import person_pos_tagger


class ProperNameCandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._loader = person_pos_tagger._load_pipeline
        person_pos_tagger._tag_text.cache_clear()

    def tearDown(self) -> None:
        person_pos_tagger._load_pipeline = self._loader
        person_pos_tagger._tag_text.cache_clear()

    def test_adjacent_proper_nouns_become_one_source_linked_candidate(self) -> None:
        def tagger(_text):
            return [
                {"entity_group": "PROPN", "start": 8, "end": 11, "score": 0.9},
                {"entity_group": "PROPN", "start": 12, "end": 17, "score": 0.8},
            ]

        person_pos_tagger._load_pipeline = lambda: tagger
        candidates, status = person_pos_tagger.proper_name_candidates(["2023-04 Ken Oyler offshore"])

        self.assertEqual(candidates, [{
            "candidate_id": "V1C1", "value_id": "V1", "text": "Ken Oyler", "start": 8,
            "end": 17, "score": 0.85, "method": "bertweet_pos_propn",
        }])
        self.assertEqual(status["candidates_found"], 1)


if __name__ == "__main__":
    unittest.main()
