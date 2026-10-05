import { createServer } from "vite";
import { spawn } from "node:child_process";
import electron from "electron";

const server = await createServer();
await server.listen();
server.printUrls();
const child = spawn(electron, ["."], {
  stdio: "inherit",
  env: { ...process.env, ASL_DEV_URL: "http://127.0.0.1:5173" },
});
let closing = false;
async function close(code = 0) {
  if (closing) return;
  closing = true;
  child.kill();
  await server.close();
  process.exit(code);
}
child.on("error", (error) => {
  console.error(error);
  void close(1);
});
child.on("exit", (code) => void close(code ?? 1));
process.on("SIGINT", () => void close());
process.on("SIGTERM", () => void close());
