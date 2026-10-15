"""Dedicated local roots keep extractor and labor artifacts separate."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
OTL_ROOT = REPO_ROOT / "orchestration-flow"
for path in (str(REPO_ROOT), str(OTL_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from otl.output_writer import make_otl_sink  # noqa: E402
from otl.sharepoint_reader import SharePointReader  # noqa: E402
from shared.output_paths import LocalSink, make_sink  # noqa: E402


class LocalSinkRootTests(unittest.TestCase):
    def test_local_sinks_use_dedicated_roots_not_the_legacy_shared_output_base(self) -> None:
        with patch.dict(os.environ, {
            "OUTPUT_TARGET": "local",
            "OUTPUT_BASE": "/legacy/shared-output",
            "OUTPUT_BASE_EXTRACTOR": "/runs/extractor/Run 1",
            "OUTPUT_BASE_PIPELINE": "/runs/labor/Run 1",
        }, clear=False):
            extractor_sink = make_sink()
            pipeline_sink = make_otl_sink()

        self.assertIsInstance(extractor_sink, LocalSink)
        self.assertEqual(Path("/runs/extractor/Run 1"), extractor_sink.base)
        self.assertIsInstance(pipeline_sink, LocalSink)
        self.assertEqual(Path("/runs/labor/Run 1"), pipeline_sink.base)

    def test_reader_defaults_to_dedicated_pipeline_input_root(self) -> None:
        with patch.dict(os.environ, {
            "INPUT_BASE": "/legacy/extractor-output",
            "INPUT_BASE_PIPELINE": "/runs/extractor/Run 1",
        }, clear=False):
            reader = SharePointReader()

        self.assertEqual("/runs/extractor/Run 1", reader.input_base)


if __name__ == "__main__":
    unittest.main()
