import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const model = path.join(root, "data", "hand_landmarker.task");
if (!existsSync(model))
  throw new Error("Run ./setup.sh to download the MediaPipe model before packaging.");
const classifier = path.join(root, "data", "knn.pkl");
const neuralModels = ["handshape.pt", "sequence.pt"]
  .map((name) => path.join(root, "data", name))
  .filter(existsSync);
if (!existsSync(classifier) && !neuralModels.length)
  throw new Error("Run uv run python model/knn.py to train the ASL classifier before packaging.");
const child = spawn(
  "uv",
  [
    "run",
    "--locked",
    ...(neuralModels.length ? ["--extra", "neural"] : []),
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
    ...(existsSync(classifier) ? ["--add-data", `${classifier}:data`] : []),
    ...neuralModels.flatMap((filename) => ["--add-data", `${filename}:data`]),
    ...(neuralModels.length ? ["--collect-all", "torch", "--hidden-import", "model.neural"] : []),
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
