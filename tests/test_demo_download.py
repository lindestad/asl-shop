"""Check credentials and interrupted/error responses without contacting Kaggle."""
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from scripts import download_demo_videos as download

MP4 = b'\x00\x00\x00\x18ftypisom' + b'video payload'


class DemoDownloadTests(unittest.TestCase):
    def test_missing_credentials_explain_setup_without_creating_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'videos'
            with patch.dict(os.environ, {'KAGGLE_CONFIG_DIR': directory}, clear=True), \
                    patch('sys.argv', ['download_demo_videos.py', '--directory', str(target)]), \
                    patch('sys.stderr', new_callable=io.StringIO) as errors:
                with self.assertRaises(SystemExit) as failure:
                    download.main()
                self.assertEqual(failure.exception.code, 1)
                self.assertIn('https://www.kaggle.com/settings/api', errors.getvalue())
                self.assertIn('kaggle.json', errors.getvalue())
                self.assertFalse(target.exists())

    def test_token_and_legacy_file_authentication(self):
        with patch.dict(os.environ, {'KAGGLE_API_TOKEN': 'secret'}, clear=True):
            self.assertEqual(download.credentials(), 'Bearer secret')
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'kaggle.json').write_text('{"username":"user","key":"key"}')
            with patch.dict(os.environ, {'KAGGLE_CONFIG_DIR': directory}, clear=True):
                self.assertEqual(download.credentials(), 'Basic dXNlcjprZXk=')

    def test_raw_and_zipped_downloads_produce_video(self):
        archive = io.BytesIO()
        with ZipFile(archive, 'w') as zip_file:
            zip_file.writestr('nested/clip.mp4', MP4)
        for payload in (MP4, archive.getvalue()):
            with self.subTest(zipped=payload != MP4), tempfile.TemporaryDirectory() as directory:
                with patch.object(download, 'urlopen', return_value=io.BytesIO(payload)):
                    download.download_clip('clip', Path(directory), 'Bearer test')
                self.assertEqual((Path(directory) / 'clip.mp4').read_bytes(), MP4)
                self.assertEqual(len(list(Path(directory).iterdir())), 1)

    def test_html_response_is_rejected_and_partial_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(download, 'urlopen', return_value=io.BytesIO(b'<html>Login</html>')):
                with self.assertRaisesRegex(ValueError, 'Kaggle API key'):
                    download.download_clip('clip', Path(directory), 'Bearer test')
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_cached_selection_does_not_require_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in download.CLIPS:
                (Path(directory) / f'{name}.mp4').write_bytes(MP4)
            with patch('sys.argv', ['download_demo_videos.py', '--directory', directory]), \
                    patch.object(download, 'credentials', side_effect=AssertionError('cached')):
                download.main()
