"""Step ordering and skipping of the one-command demo setup; never contacts Kaggle or Drive."""
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from scripts import demo_replay, setup_demo_replay as setup

MP4 = b'\x00\x00\x00\x18ftypisom' + b'video payload'
JSON = b'[{"clipFilename": "a.mp4"}]'


class DemoReplaySetupTests(unittest.TestCase):
    def test_missing_credentials_stop_before_any_download(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            with patch.dict(os.environ, {'KAGGLE_CONFIG_DIR': str(directory)}, clear=True), \
                    patch.object(demo_replay, 'PACKED', directory / 'packed'), \
                    patch.object(demo_replay, 'CLIP_VIDEOS', directory / 'videos'), \
                    patch.object(setup, 'download_landmarks', side_effect=AssertionError('downloaded')):
                with self.assertRaises(SystemExit) as failure:
                    setup.main(openfs=directory / 'openfs')
            self.assertIn('https://www.kaggle.com/settings/api', str(failure.exception.code))
            self.assertEqual(list(directory.iterdir()), [])

    def test_metadata_accepts_raw_and_zipped_and_rejects_html(self):
        archive = io.BytesIO()
        with ZipFile(archive, 'w') as zip_file:
            zip_file.writestr('nested/meta.json', JSON)
        for payload in (JSON, archive.getvalue()):
            with self.subTest(zipped=payload != JSON), tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / 'meta.json'
                with patch.object(setup, 'urlopen', return_value=io.BytesIO(payload)):
                    setup.fetch_kaggle_file('x/meta.json', target, 'Bearer test')
                self.assertEqual(target.read_bytes(), JSON)
                self.assertEqual(len(list(Path(directory).iterdir())), 1)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(setup, 'urlopen', return_value=io.BytesIO(b'<html>Login</html>')):
                with self.assertRaisesRegex(ValueError, 'Kaggle API key'):
                    setup.fetch_kaggle_file('x/meta.json', Path(directory) / 'meta.json', 'Bearer test')
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_finished_setup_needs_no_credentials_and_downloads_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / 'packed').mkdir()
            (directory / 'packed' / 'test_clips.json').write_text('[]')
            (directory / 'videos').mkdir()
            for name in setup.videos.CLIPS:
                (directory / 'videos' / f'{name}.mp4').write_bytes(MP4)
            used = AssertionError('should be skipped')
            with patch.object(demo_replay, 'PACKED', directory / 'packed'), \
                    patch.object(demo_replay, 'CLIP_VIDEOS', directory / 'videos'), \
                    patch.object(setup.videos, 'credentials', side_effect=used), \
                    patch.object(setup, 'download_metadata', side_effect=used), \
                    patch.object(setup, 'download_landmarks', side_effect=used), \
                    patch.object(setup, 'pack', side_effect=used), \
                    patch.object(setup, 'verify') as verify, \
                    patch('sys.stdout', new_callable=io.StringIO):
                setup.main(openfs=directory / 'openfs')
            verify.assert_called_once()

    def test_interrupted_landmark_download_is_resumed(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(setup.download_openfs, 'fetch_archive',
                              side_effect=[ValueError('broken pipe'), None]) as fetch, \
                    patch('sys.stdout', new_callable=io.StringIO):
                setup.download_landmarks(Path(directory))
            self.assertEqual(fetch.call_count, 2)


if __name__ == '__main__':
    unittest.main()
