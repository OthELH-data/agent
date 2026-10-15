"""Tests for otl/jv_object_types.py — the major-construction JV object-type code registry."""
from __future__ import annotations

import os
import unittest

from otl import jv_object_types

_EXPECTED_CODES = {
    "C116", "C136", "C141", "C142", "C143", "C144", "C145",
    "O117", "O135", "O142", "O143", "O144", "O145", "O153",
}


class RegistryLoadTests(unittest.TestCase):

    def setUp(self):
        jv_object_types.major_construction_jv_object_types.cache_clear()

    def tearDown(self):
        jv_object_types.major_construction_jv_object_types.cache_clear()

    def test_returns_all_fourteen_codes(self):
        codes = jv_object_types.major_construction_jv_object_types()
        self.assertEqual(set(codes), _EXPECTED_CODES)
        self.assertEqual(len(codes), 14)

    def test_descriptions_are_non_empty_strings(self):
        codes = jv_object_types.major_construction_jv_object_types()
        for code, description in codes.items():
            self.assertIsInstance(description, str)
            self.assertTrue(description.strip(), f"{code} has an empty description")

    def test_result_is_cached(self):
        first = jv_object_types.major_construction_jv_object_types()
        second = jv_object_types.major_construction_jv_object_types()
        self.assertIs(first, second)


class NormalizationTests(unittest.TestCase):

    def setUp(self):
        jv_object_types.major_construction_jv_object_types.cache_clear()

    def tearDown(self):
        jv_object_types.major_construction_jv_object_types.cache_clear()

    def test_is_major_construction_code_matches_exact(self):
        self.assertTrue(jv_object_types.is_major_construction_code("C142"))

    def test_matches_regardless_of_case(self):
        self.assertTrue(jv_object_types.is_major_construction_code("c142"))

    def test_matches_with_surrounding_whitespace(self):
        self.assertTrue(jv_object_types.is_major_construction_code("  C142  "))

    def test_matches_with_internal_whitespace(self):
        self.assertTrue(jv_object_types.is_major_construction_code("C 142"))

    def test_matches_site_prefixed_code(self):
        self.assertTrue(jv_object_types.is_major_construction_code("US01/C116"))

    def test_unlisted_code_is_not_a_match(self):
        self.assertFalse(jv_object_types.is_major_construction_code("C999"))

    def test_empty_value_is_not_a_match(self):
        self.assertFalse(jv_object_types.is_major_construction_code(""))

    def test_none_value_is_not_a_match(self):
        self.assertFalse(jv_object_types.is_major_construction_code(None))


class RegistryPathOverrideTests(unittest.TestCase):

    def setUp(self):
        jv_object_types.major_construction_jv_object_types.cache_clear()
        self._prior = os.environ.pop("OTL_MAJOR_CONSTRUCTION_REGISTRY", None)

    def tearDown(self):
        jv_object_types.major_construction_jv_object_types.cache_clear()
        if self._prior is not None:
            os.environ["OTL_MAJOR_CONSTRUCTION_REGISTRY"] = self._prior
        else:
            os.environ.pop("OTL_MAJOR_CONSTRUCTION_REGISTRY", None)

    def test_missing_registry_file_raises(self):
        os.environ["OTL_MAJOR_CONSTRUCTION_REGISTRY"] = "/nonexistent/path/registry.yaml"
        with self.assertRaises(FileNotFoundError):
            jv_object_types.major_construction_jv_object_types()

    def test_malformed_registry_raises_value_error(self, tmp_path=None):
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("codes: not-a-list\n")
            path = f.name
        try:
            os.environ["OTL_MAJOR_CONSTRUCTION_REGISTRY"] = path
            with self.assertRaises(ValueError):
                jv_object_types.major_construction_jv_object_types()
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
