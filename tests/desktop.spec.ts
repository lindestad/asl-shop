import { test, expect, _electron as electron } from "@playwright/test";
import { mkdtemp, readFile, readdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";

test("Electron connects native image import, real Python recognition, and batch folder export", async () => {
  const output = await mkdtemp(path.join(tmpdir(), "asl-shop-test-"));
  const application = await electron.launch({
    executablePath: process.env.ASL_SHOP_TEST_APP,
    args: process.env.ASL_SHOP_TEST_APP ? [] : ["."],
    env: { ...process.env, ASL_DEV_URL: "http://127.0.0.1:5173" },
  });
  try {
    const page = await application.firstWindow();
    await page.waitForLoadState("domcontentloaded");
    const sample = path.resolve("data/asl_alphabet_train/asl_alphabet_train/A/A1000.jpg");
    await application.evaluate(({ dialog }, filename) => {
      dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [filename] });
    }, sample);
    await page.getByRole("button", { name: "Open image", exact: true }).first().click();
    await expect(page.getByTestId("editor-canvas")).toBeVisible();
    const isolated = await page.evaluate(() => ({
      node: typeof (window as unknown as { require?: unknown }).require,
      bridge: typeof window.desktop?.recognize,
    }));
    expect(isolated).toEqual({ node: "undefined", bridge: "function" });
    await page.getByRole("button", { name: "Recognize", exact: true }).click();
    await page.getByRole("button", { name: "Analyze image", exact: true }).click();
    await expect(page.locator(".hand-result")).toHaveCount(1, { timeout: 45000 });
    await expect(
      page.locator(".editor-workspace").getByText("Letter classifier unavailable", { exact: true }),
    ).toBeVisible();
    const destination = path.join(output, "edited.png");
    await application.evaluate(({ dialog }, filename) => {
      dialog.showSaveDialog = async () => ({ canceled: false, filePath: filename });
    }, destination);
    await page.getByRole("button", { name: "Export", exact: true }).click();
    await expect(page.getByRole("status")).toContainText("Image exported");
    expect((await readFile(destination)).subarray(0, 8).toString("hex")).toBe("89504e470d0a1a0a");
    await page.getByRole("button", { name: "Batch studio", exact: true }).click();
    await page.getByRole("button", { name: "Add images", exact: true }).last().click();
    await page.getByRole("button", { name: "Analyze all images", exact: true }).click();
    await expect(page.locator(".batch-status.done")).toHaveCount(1);
    await application.evaluate(({ dialog }, directory) => {
      dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [directory] });
    }, output);
    await page.getByRole("button", { name: "Choose output folder", exact: true }).click();
    await expect(page.getByRole("status")).toContainText("Exported 1 images");
    const batchDirectory = (await readdir(output)).find((entry) => entry.startsWith("asl-shop-"))!;
    const report = JSON.parse(
      await readFile(path.join(output, batchDirectory, "recognition.json"), "utf8"),
    );
    expect(report[0].recognition.hands[0].features).toHaveLength(63);
    expect(report[0].recognition.prediction).toBeNull();
  } finally {
    await application.close();
    await rm(output, { recursive: true, force: true });
  }
});

test("camera tracking keeps up with video, finds additional hands, and restarts cleanly", async () => {
  const application = await electron.launch({
    executablePath: process.env.ASL_SHOP_TEST_APP,
    args: process.env.ASL_SHOP_TEST_APP ? [] : ["."],
    env: { ...process.env, ASL_DEV_URL: "http://127.0.0.1:5173" },
  });
  try {
    const page = await application.firstWindow();
    await page.waitForLoadState("domcontentloaded");
    const sample = await readFile("data/asl_alphabet_train/asl_alphabet_train/A/A1000.jpg");
    await page.evaluate(async (base64) => {
      const image = new Image();
      image.src = `data:image/jpeg;base64,${base64}`;
      await image.decode();
      const canvas = document.createElement("canvas");
      canvas.width = 640;
      canvas.height = 480;
      const context = canvas.getContext("2d")!;
      const draw = () => {
        context.drawImage(image, 0, 0, 640, 480);
        // A changing pixel ensures Chromium presents a fresh video frame.
        context.fillStyle = `rgb(${Math.floor(performance.now()) % 255}, 0, 0)`;
        context.fillRect(0, 0, 1, 1);
      };
      draw();
      setInterval(draw, 1000 / 30);
      navigator.mediaDevices.getUserMedia = async () => canvas.captureStream(30);
    }, sample.toString("base64"));
    await page.getByRole("button", { name: "Live camera", exact: true }).click();
    await page.getByRole("button", { name: "Start camera", exact: true }).first().click();
    const camera = page.locator(".workspace:not([hidden])");
    await expect(camera.locator(".hand-result")).toHaveCount(1, { timeout: 45000 });
    // The old 180 ms pause capped updates below 6 FPS even on a fast machine.
    await expect
      .poll(async () =>
        parseInt((await page.getByLabel("Tracking frame rate").textContent()) || "0"),
      )
      .toBeGreaterThan(10);
    await test.info().attach("tracking frame rate", {
      body: (await page.getByLabel("Tracking frame rate").textContent())!,
      contentType: "text/plain",
    });

    await page.getByRole("button", { name: "Stop camera", exact: true }).click();
    await expect(page.getByLabel("Tracking frame rate")).toHaveCount(0);
    const multiple = await page.evaluate(async (base64) => {
      const image = new Image();
      image.src = `data:image/jpeg;base64,${base64}`;
      await image.decode();
      const canvas = document.createElement("canvas");
      canvas.width = canvas.height = 400;
      const context = canvas.getContext("2d")!;
      for (let y = 0; y < 2; y++)
        for (let x = 0; x < 2; x++) context.drawImage(image, x * 200, y * 200, 200, 200);
      return canvas.toDataURL("image/jpeg", 0.95);
    }, sample.toString("base64"));
    await expect
      .poll(async () =>
        page.evaluate(
          async (dataUrl) => (await window.desktop!.recognizeFrame(dataUrl)).hands.length,
          multiple,
        ),
      )
      .toBeGreaterThan(2);
    const still = await page.evaluate(
      async (dataUrl) => window.desktop!.recognize(dataUrl),
      multiple,
    );
    expect(still.hands.length).toBeGreaterThan(2);
    for (const hand of still.hands) {
      expect(hand.landmarks).toHaveLength(21);
      expect(hand.features).toHaveLength(63);
    }
    await page.getByRole("button", { name: "Start camera", exact: true }).first().click();
    await expect(camera.locator(".hand-result")).toHaveCount(1);
    await page.getByRole("button", { name: "Capture frame" }).click();
    await expect(page.getByTestId("editor-canvas")).toBeVisible();
    expect(
      await page.locator("video").evaluate((element) => (element as HTMLVideoElement).srcObject),
    ).toBeNull();
  } finally {
    await application.close();
  }
});
