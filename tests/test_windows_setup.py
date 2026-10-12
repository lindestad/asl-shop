"""Exercise native setup argument handling without installing or downloading."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")


@unittest.skipUnless(POWERSHELL, "PowerShell is required for Windows setup CLI checks")
class WindowsSetupTests(unittest.TestCase):
    def plan(self, *arguments):
        result = subprocess.run([POWERSHELL, "-NoProfile", "-File", str(ROOT / "setup.ps1"),
                                 "-Plan", *arguments], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_default_setup_does_not_download_openfs_or_install_torch(self):
        plan = self.plan()
        self.assertFalse(plan["neural"])
        self.assertFalse(plan["openfs"])
        self.assertFalse(plan["extract"])
        self.assertNotIn("neural", plan["pythonSync"])
        self.assertFalse(plan["openfsCommand"])

    def test_neural_setup_is_independent_of_data_download(self):
        plan = self.plan("-Neural")
        self.assertTrue(plan["neural"])
        self.assertFalse(plan["openfs"])
        self.assertEqual(plan["pythonSync"][-2:], ["--extra", "neural"])

    def test_extraction_preserves_custom_path_with_spaces_and_implies_neural(self):
        destination = ROOT / "data" / "external" / "data with spaces"
        plan = self.plan("-ExtractOpenFS", "-OpenFSDir", str(destination))
        self.assertTrue(plan["neural"])
        self.assertTrue(plan["openfs"])
        self.assertTrue(plan["extract"])
        self.assertEqual(plan["openfsDirectory"], str(destination))
        self.assertEqual(plan["openfsCommand"][-3:], ["--directory", str(destination), "--extract"])
        self.assertFalse(destination.exists())

    def test_download_option_does_not_implicitly_extract(self):
        plan = self.plan("-OpenFS")
        self.assertTrue(plan["openfs"])
        self.assertFalse(plan["extract"])
        self.assertNotIn("--extract", plan["openfsCommand"])

    def test_neural_launch_plan_keeps_gpu_training_and_runtime_paths_independent(self):
        python_path = ROOT / ".venv" / "Scripts" / "python.exe"
        checkpoint_path = ROOT / "data" / "model experiment" / "handshape.pt"
        result = subprocess.run([POWERSHELL, "-NoProfile", "-File", str(ROOT / "scripts/start-neural.ps1"),
                                 "-Plan", "-Python", str(python_path), "-HandshapeModel", str(checkpoint_path)],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["environment"]["ASL_SHOP_PYTHON"], str(python_path))
        self.assertEqual(plan["environment"]["ASL_HANDSHAPE_MODEL"], str(checkpoint_path))
        self.assertNotIn("training-env", plan["environment"]["ASL_SHOP_PYTHON"])
        self.assertEqual(plan["command"], ["npm.cmd", "run", "dev"])

    def test_openfs_launch_requires_explicit_source_and_checkpoint_pair(self):
        launcher = [POWERSHELL, "-NoProfile", "-File", str(ROOT / "scripts/start-neural.ps1"), "-Plan"]
        result = subprocess.run([*launcher, "-OpenFSSource", ".build/openfs"],
                                capture_output=True, text=True, timeout=15)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires both", result.stderr)
        result = subprocess.run([*launcher, "-OpenFSSource", ".build/openfs", "-OpenFSModelDir", "data/openfs-checkpoints/recognizer_fsboard"],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["environment"]["ASL_OPENFS_SOURCE"], str(ROOT / ".build" / "openfs"))
        self.assertEqual(plan["environment"]["ASL_OPENFS_MODEL_DIR"], str(ROOT / "data" / "openfs-checkpoints" / "recognizer_fsboard"))


if __name__ == "__main__":
    unittest.main()
