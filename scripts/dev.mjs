import { createServer } from "vite";
import { spawn } from "node:child_process";
import { watch } from "node:fs";
import path from "node:path";
import electron from "electron";

const server = await createServer();
await server.listen();
server.printUrls();
let closing = false;
let restarting = false;
let restartTimer;
let child;

function startElectron() {
  child = spawn(electron, ["."], {
    stdio: "inherit",
    env: { ...process.env, ASL_DEV_URL: "http://127.0.0.1:5173" },
  });
  child.on("error", (error) => {
    console.error(error);
    void close(1);
  });
  child.on("exit", (code) => {
    if (closing) return;
    if (restarting) {
      restarting = false;
      startElectron();
    } else {
      void close(code ?? 1);
    }
  });
}

const watcher = watch(path.resolve("electron"), { recursive: true }, (_, filename) => {
  if (!filename || !/\.(?:mjs|cjs)$/.test(filename)) return;
  clearTimeout(restartTimer);
  restartTimer = setTimeout(() => {
    if (closing || restarting) return;
    restarting = true;
    child.kill();
  }, 150);
});
watcher.on("error", (error) => {
  console.error(error);
  void close(1);
});

async function close(code = 0) {
  if (closing) return;
  closing = true;
  clearTimeout(restartTimer);
  watcher.close();
  child?.kill();
  await server.close();
  process.exit(code);
}
startElectron();
process.on("SIGINT", () => void close());
process.on("SIGTERM", () => void close());
