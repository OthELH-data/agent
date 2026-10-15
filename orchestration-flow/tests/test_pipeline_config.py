from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import otl  # noqa: F401  — bootstrap repository-root shared imports

from shared.pipeline_config import (
    clear_pipeline_config_cache,
    node_gate_enabled,
    pipeline_config,
)
from otl.input_sources import extractor_excel_parsing_enabled


class PipelineConfigTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_pipeline_config_cache()

    def _config(self, text: str):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "otl_pipeline.yaml"
        path.write_text(text, encoding="utf-8")
        return directory, path

    def test_supports_expanded_and_boolean_gate_entries(self) -> None:
        directory, path = self._config(
            """schema_version: 1
node_gates:
  verify_evidence:
    enabled: false
    description: Skip the evidence gate
  future_gate: true
"""
        )
        self.addCleanup(directory.cleanup)

        with patch.dict(os.environ, {"OTL_PIPELINE_CONFIG": str(path)}):
            clear_pipeline_config_cache()
            self.assertFalse(node_gate_enabled("verify_evidence"))
            self.assertTrue(node_gate_enabled("future_gate"))
            self.assertFalse(node_gate_enabled("not_configured"))
            self.assertTrue(node_gate_enabled("not_configured", default=True))

    def test_rejects_non_boolean_enabled_values(self) -> None:
        directory, path = self._config(
            """schema_version: 1
node_gates:
  verify_evidence:
    enabled: yes-please
"""
        )
        self.addCleanup(directory.cleanup)

        with patch.dict(os.environ, {"OTL_PIPELINE_CONFIG": str(path)}):
            clear_pipeline_config_cache()
            with self.assertRaisesRegex(ValueError, "must be true or false"):
                pipeline_config()

    def test_rejects_unknown_schema_versions(self) -> None:
        directory, path = self._config("schema_version: 99\nnode_gates: {}\n")
        self.addCleanup(directory.cleanup)

        with patch.dict(os.environ, {"OTL_PIPELINE_CONFIG": str(path)}):
            clear_pipeline_config_cache()
            with self.assertRaisesRegex(ValueError, "schema_version"):
                pipeline_config()

    def test_excel_parsing_supports_expanded_and_boolean_entries(self) -> None:
        configs = (
            ("""schema_version: 1
input_sources:
  extractor_outputs:
    enabled: true
    excel_parsing:
      enabled: false
node_gates: {}
""", False),
            ("""schema_version: 1
input_sources:
  extractor_outputs:
    enabled: true
    excel_parsing: true
node_gates: {}
""", True),
        )
        for config_text, expected in configs:
            with self.subTest(expected=expected):
                directory, path = self._config(config_text)
                self.addCleanup(directory.cleanup)

                with patch.dict(os.environ, {"OTL_PIPELINE_CONFIG": str(path)}):
                    clear_pipeline_config_cache()
                    self.assertIs(extractor_excel_parsing_enabled(), expected)

    def test_excel_parsing_defaults_enabled_when_absent(self) -> None:
        directory, path = self._config(
            """schema_version: 1
input_sources:
  extractor_outputs:
    enabled: true
node_gates: {}
"""
        )
        self.addCleanup(directory.cleanup)

        with patch.dict(os.environ, {"OTL_PIPELINE_CONFIG": str(path)}):
            clear_pipeline_config_cache()
            self.assertTrue(extractor_excel_parsing_enabled())

    def test_rejects_invalid_excel_parsing_switch(self) -> None:
        directory, path = self._config(
            """schema_version: 1
input_sources:
  extractor_outputs:
    enabled: true
    excel_parsing:
      enabled: yes-please
node_gates: {}
"""
        )
        self.addCleanup(directory.cleanup)

        with patch.dict(os.environ, {"OTL_PIPELINE_CONFIG": str(path)}):
            clear_pipeline_config_cache()
            with self.assertRaisesRegex(
                ValueError,
                r"extractor_outputs\.excel_parsing\.enabled must be true or false",
            ):
                pipeline_config()


if __name__ == "__main__":
    unittest.main()
