import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const model = path.join(root, "data", "hand_landmarker.task");
if (!existsSync(model))
  throw new Error("Run ./setup.sh to download the MediaPipe model before packaging.");
const trainedModels = ["knn_model.pkl", "cnn-landmarks.pkl"].map((name) =>
  path.join(root, "model", "trained_models", name),
);
for (const classifier of trainedModels)
  if (!existsSync(classifier)) throw new Error(`Missing trained model: ${classifier}`);
const neuralModels = ["handshape.pt", "sequence.pt"].map((name) => path.join(root, "data", name));
for (const checkpoint of neuralModels)
  if (!existsSync(checkpoint)) throw new Error(`Missing neural model: ${checkpoint}`);
const ensembleDirectory = "data/models/ensemble-2026-10-25-bf16";
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
    ...trainedModels.flatMap((filename) => ["--add-data", `${filename}:model/trained_models`]),
    ...neuralModels.flatMap((filename) => ["--add-data", `${filename}:data`]),
    "--add-data",
    `${path.join(root, ensembleDirectory)}:${ensembleDirectory}`,
    "--collect-all",
    "torch",
    "--hidden-import",
    "model.neural",
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
