from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from artifacts import (
    C1_MANIFEST_NAME,
    C1_OUTPUTS,
    build_c1_manifest,
    publish_directory,
    replace_artifact,
    staging_directory,
    validate_c1_manifest,
    write_json_atomic,
)


class ArtifactTest(unittest.TestCase):
    def metadata(self) -> dict:
        return {
            "run_id": "run",
            "created_at_utc": "now",
            "producer": {"commit": "a" * 40},
            "parent_prepared_manifest": {"sha256": "b" * 64},
            "input_path": "/input",
            "input_data_sha256": "c" * 64,
            "input_schema_sha256": "d" * 64,
            "row_count": 1,
        }

    def populate(self, output: Path, metadata: dict) -> None:
        for relative in C1_OUTPUTS.values():
            path = output / relative
            if path.suffix:
                path.write_text(relative, encoding="utf-8")
            else:
                path.mkdir()
                (path / "part").write_text(relative, encoding="utf-8")
        write_json_atomic(metadata, output / "audio_encoder_metadata.json")
        manifest = build_c1_manifest(output, metadata)
        write_json_atomic(manifest, output / C1_MANIFEST_NAME)

    def test_publish_replaces_the_whole_old_directory(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "audio"
            output.mkdir()
            (output / "stale.faiss").write_text("stale", encoding="utf-8")
            staging = staging_directory(output, "run")
            (staging / "new").write_text("new", encoding="utf-8")
            publish_directory(staging, output, "run")
            self.assertFalse((output / "stale.faiss").exists())
            self.assertEqual((output / "new").read_text(encoding="utf-8"), "new")

    def test_manifest_detects_output_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            output = Path(root)
            metadata = self.metadata()
            self.populate(output, metadata)
            validate_c1_manifest(output, metadata)
            (output / "pca_model" / "part").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "pca_model hash mismatch"):
                validate_c1_manifest(output, metadata)

    def test_failed_artifact_replace_restores_target(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "target"
            target.write_text("old", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                replace_artifact(Path(root) / "missing", target, "run")
            self.assertEqual(target.read_text(encoding="utf-8"), "old")


if __name__ == "__main__":
    unittest.main()
