"""Bundled weights must resolve independently of the launch directory."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from model.ensemble_refiner import DEFAULT_MODEL_DIR, EnsembleRefiner
from model.live_neural import NeuralRuntime

ROOT = Path(__file__).resolve().parents[1]


class EnsembleDefaultTests(unittest.TestCase):
    def test_repository_default_has_all_four_members_without_export(self):
        with patch.dict(os.environ, {}, clear=True):
            refiner = EnsembleRefiner()
        self.assertEqual(refiner.directory, ROOT / DEFAULT_MODEL_DIR)
        self.assertEqual(refiner.readiness()['members'], 4)
        self.assertEqual(refiner.readiness()['status'], 'configured')

    def test_runtime_uses_its_resource_root(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {}, clear=True):
            runtime = NeuralRuntime(root)
            self.assertEqual(runtime.ensemble.directory, Path(root) / DEFAULT_MODEL_DIR)

    def test_explicit_path_overrides_environment_and_default(self):
        with patch.dict(os.environ, {'ASL_ENSEMBLE_MODEL_DIR': '/custom/weights'}):
            self.assertEqual(EnsembleRefiner().directory, '/custom/weights')
            self.assertEqual(EnsembleRefiner('/explicit/weights').directory, '/explicit/weights')
