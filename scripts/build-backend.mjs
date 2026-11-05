import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const model = path.join(root, "data", "hand_landmarker.task");
if (!existsSync(model))
  throw new Error("Run ./setup.sh to download the MediaPipe model before packaging.");
const classifiers = ["knn_model.pkl", "cnn-landmarks.pkl"].map((name) =>
  path.join(root, "model", "trained_models", name),
);
for (const classifier of classifiers)
  if (!existsSync(classifier)) throw new Error(`Missing trained model: ${classifier}`);
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
    ...classifiers.flatMap((classifier) => ["--add-data", `${classifier}:model/trained_models`]),
    "--collect-all",
    "mediapipe",
    "--hidden-import",
    "handTracker.hand_features",
    "--collect-submodules",
    "sklearn.neighbors",
    "scripts/recognition_worker.py",
  ],
  { cwd: root, stdio: "inherit" },
);
child.on("error", (error) => {
  console.error(error.message);
  process.exit(1);
});
child.on("exit", (code) => process.exit(code ?? 1));
