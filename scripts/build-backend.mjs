import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const model = path.join(root, "data", "hand_landmarker.task");
if (!existsSync(model))
  throw new Error("Run ./setup.sh to download the MediaPipe model before packaging.");
const classifier = path.join(root, "model", "asl_knn.npz");
if (!existsSync(classifier))
  throw new Error("Run uv run python model/knn.py to train the ASL classifier before packaging.");
const child = spawn(
  "uv",
  [
    "run",
    "--locked",
    "--python",
    "3.12",
    "--with",
    "pyinstaller==6.22.3",
    "pyinstaller",
    "--noconfirm",
    "--onedir",
    "--name",
    "asl-recognition",
    "--distpath",
    ".build/backend",
    "--workpath",
    ".build/work",
    "--specpath",
    ".build",
    "--paths",
    root,
    "--add-data",
    `${model}:data`,
    "--add-data",
    `${classifier}:model`,
    "--collect-all",
    "mediapipe",
    "--hidden-import",
    "handTracker.hand_features",
    "scripts/recognition_worker.py",
  ],
  { cwd: root, stdio: "inherit" },
);
child.on("error", (error) => {
  console.error(error.message);
  process.exit(1);
});
child.on("exit", (code) => process.exit(code ?? 1));
