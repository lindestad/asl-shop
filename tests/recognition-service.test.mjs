import assert from "node:assert/strict";
import test from "node:test";
import { RecognitionService } from "../electron/recognition.mjs";

test("invalid stream identity and timestamps are rejected before starting a worker", async () => {
  const service = new RecognitionService("unused");
  const image = "data:image/png;base64,eA==";
  for (const options of [
    {},
    { sessionId: "one", timestampMs: NaN },
    { sessionId: "one", timestampMs: 1, finish: "yes" },
  ]) {
    await assert.rejects(service.analyze(image, true, options), /session ID and capture timestamp/);
  }
  service.stop();
});
