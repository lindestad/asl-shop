import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { existsSync } from "node:fs";
import path from "node:path";

export class RecognitionService {
  #child = null;
  #pending = new Map();
  #nextId = 0;

  constructor(root, packaged = false) {
    this.root = root;
    this.packaged = packaged;
  }

  #start() {
    let executable;
    let args;
    if (this.packaged) {
      executable = path.join(
        this.root,
        "backend",
        "asl-recognition",
        process.platform === "win32" ? "asl-recognition.exe" : "asl-recognition",
      );
      args = [];
    } else {
      const venv = path.join(
        this.root,
        ".venv",
        process.platform === "win32" ? "Scripts/python.exe" : "bin/python",
      );
      executable = process.env.ASL_SHOP_PYTHON || (existsSync(venv) ? venv : "python3");
      args = [path.join(this.root, "scripts", "recognition_worker.py")];
    }
    const child = spawn(executable, args, {
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true,
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    });
    this.#child = child;
    let stderr = "";
    child.stderr.on("data", (chunk) => {
      stderr = (stderr + chunk).slice(-4000);
    });
    const fail = (error) => {
      if (this.#child !== child) return;
      this.#child = null;
      for (const request of this.#pending.values()) {
        clearTimeout(request.timer);
        request.reject(error);
      }
      this.#pending.clear();
    };
    child.on("error", () =>
      fail(new Error("Could not start hand detection. Run ./setup.sh first.")),
    );
    child.on("exit", (code) =>
      fail(
        new Error(
          `Hand detection stopped (${code ?? "signal"}). ${stderr.includes("ModuleNotFoundError") ? "Run ./setup.sh to install the Python dependencies." : "Check that the MediaPipe model is installed."}`,
        ),
      ),
    );
    child.stdin.on("error", (error) => fail(error));
    createInterface({ input: child.stdout }).on("line", (line) => {
      let message;
      try {
        message = JSON.parse(line);
      } catch {
        return;
      }
      const pending = this.#pending.get(message.id);
      if (!pending) return;
      clearTimeout(pending.timer);
      this.#pending.delete(message.id);
      if (message.error) pending.reject(new Error(message.error));
      else pending.resolve(message.result);
    });
  }

  analyze(dataUrl, video = false) {
    if (
      typeof dataUrl !== "string" ||
      dataUrl.length > 24 * 1024 * 1024 ||
      !/^data:image\/(png|jpeg|webp);base64,/.test(dataUrl)
    ) {
      return Promise.reject(
        new Error("Hand detection requires a PNG, JPEG, or WebP image under 18 MB."),
      );
    }
    if (!this.#child) this.#start();
    const id = ++this.#nextId;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.stop(new Error("Hand detection timed out. Try again with a smaller image."));
      }, 60000);
      this.#pending.set(id, { resolve, reject, timer });
      this.#child.stdin.write(JSON.stringify({ id, dataUrl, video }) + "\n");
    });
  }

  stop(error = new Error("Hand detection was closed.")) {
    for (const request of this.#pending.values()) {
      clearTimeout(request.timer);
      request.reject(error);
    }
    this.#pending.clear();
    const child = this.#child;
    this.#child = null;
    child?.kill();
  }
}
