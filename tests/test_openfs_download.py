"""Local archive behavior; these tests never contact Google Drive."""
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from scripts.download_openfs import archive_ready, extract_archive
from scripts.validate_landmarker import promote_model


class OpenFSDownloadTests(unittest.TestCase):
    def test_landmarker_promotion_validates_before_replacing(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            destination = directory / "hand_landmarker.task"
            destination.write_bytes(b"existing")
            partial = directory / "hand_landmarker.task.part"
            with ZipFile(partial, "w"):
                pass
            with self.assertRaisesRegex(ValueError, "empty or corrupt"):
                promote_model(partial)
            self.assertEqual(destination.read_bytes(), b"existing")
            with ZipFile(partial, "w") as archive:
                archive.writestr("model.tflite", b"model")
            promote_model(partial)
            self.assertFalse(partial.exists())
            self.assertTrue(archive_ready(destination))

    def test_existing_archive_extracts_without_removing_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "data.zip"
            with ZipFile(source, "w") as archive:
                archive.writestr("fingerspelling_fsboard_raw/train.npz", b"sample")
            self.assertTrue(archive_ready(source))
            destination = directory / "extracted"
            extract_archive(source, destination)
            self.assertEqual((destination / "fingerspelling_fsboard_raw/train.npz").read_bytes(), b"sample")
            self.assertTrue(source.is_file())
            extract_archive(source, destination)

    def test_refuses_path_escape_before_writing_any_members(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "unsafe.zip"
            with ZipFile(source, "w") as archive:
                archive.writestr("okay.npz", b"sample")
                archive.writestr("../outside.npz", b"outside")
            destination = directory / "extracted"
            with self.assertRaisesRegex(ValueError, "leaves the extraction directory"):
                extract_archive(source, destination)
            self.assertFalse((directory / "outside.npz").exists())
            self.assertFalse((destination / "okay.npz").exists())

    def test_refuses_to_overwrite_different_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "data.zip"
            with ZipFile(source, "w") as archive:
                archive.writestr("train.npz", b"sample")
            destination = directory / "extracted"
            destination.mkdir()
            existing = destination / "train.npz"
            existing.write_bytes(b"keep this")
            with self.assertRaisesRegex(ValueError, "different size"):
                extract_archive(source, destination)
            self.assertEqual(existing.read_bytes(), b"keep this")


if __name__ == "__main__":
    unittest.main()
