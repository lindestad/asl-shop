import { test, expect, _electron as electron } from "@playwright/test";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import path from "node:path";

const sampleImage = path.resolve(
  process.env.ASL_SHOP_TEST_DATA_ROOT || "data",
  "asl_alphabet_train/asl_alphabet_train/A/A1000.jpg",
);

declare global {
  interface Window {
    cameraTest: {
      multiple: boolean;
      tracks: MediaStreamTrack[];
    };
  }
}

test.beforeEach(() => {
  const sample = sampleImage;
  test.skip(!existsSync(sample), "Download the ASL dataset to run the desktop integration tests.");
  if (!process.env.ASL_SHOP_TEST_APP) {
    expect(
      existsSync(path.resolve("data/knn.pkl")) ||
        existsSync(path.resolve(process.env.ASL_HANDSHAPE_MODEL || "data/handshape.pt")),
      "Train a neural handshape model or the KNN classifier first.",
    ).toBe(true);
  }
});

test("Electron recognizes a letter from an image with the trained model", async () => {
  const sample = sampleImage;
  test.skip(!existsSync(sample), "Download the ASL dataset to run the desktop integration test.");
  const application = await electron.launch({
    executablePath: process.env.ASL_SHOP_TEST_APP,
    args: process.env.ASL_SHOP_TEST_APP ? ["--disable-gpu"] : ["--disable-gpu", "."],
    env: { ...process.env, ASL_DEV_URL: "http://127.0.0.1:5173" },
  });
  try {
    const page = await application.firstWindow();
    await expect(page.getByRole("heading", { name: "Live kamera" })).toBeVisible({
      timeout: 45000,
    });
    await application.evaluate(({ dialog }, filename) => {
      dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [filename] });
    }, sample);

    await page.getByRole("button", { name: "Bilde" }).click();
    await page.getByRole("button", { name: "Velg bilde" }).click();
    await expect(page.getByRole("img", { name: "A1000.jpg" })).toBeVisible();
    await expect(page.getByText("1 hånd funnet")).toBeVisible({ timeout: 45000 });
    await expect(page.locator(".result-letter")).toHaveText("A");
    await expect(
      page.getByText(/5 av 5 nærmeste treningseksempler|Nevral bokstavmodell/),
    ).toBeVisible();
    expect(
      await page.evaluate(() => typeof (window as unknown as { require?: unknown }).require),
    ).toBe("undefined");
    expect(await page.evaluate(() => typeof window.desktop?.recognize)).toBe("function");
  } finally {
    await application.close();
  }
});

test("camera recognizes letters, keeps up with video, finds new hands, and restarts", async () => {
  const sample = sampleImage;
  test.skip(!existsSync(sample), "Download the ASL dataset to run the desktop integration test.");
  const application = await electron.launch({
    executablePath: process.env.ASL_SHOP_TEST_APP,
    args: process.env.ASL_SHOP_TEST_APP ? ["--disable-gpu"] : ["--disable-gpu", "."],
    env: { ...process.env, ASL_DEV_URL: "http://127.0.0.1:5173" },
  });
  try {
    const page = await application.firstWindow();
    await expect(page.getByRole("heading", { name: "Live kamera" })).toBeVisible({
      timeout: 45000,
    });
    const base64 = (await readFile(sample)).toString("base64");
    await page.evaluate(async (encoded) => {
      const image = new Image();
      image.src = `data:image/jpeg;base64,${encoded}`;
      await image.decode();
      const canvas = document.createElement("canvas");
      canvas.width = 480;
      canvas.height = 480;
      const context = canvas.getContext("2d")!;
      window.cameraTest = { multiple: false, tracks: [] };
      const draw = () => {
        if (window.cameraTest.multiple) {
          for (let y = 0; y < 2; y++)
            for (let x = 0; x < 2; x++) context.drawImage(image, x * 240, y * 240, 240, 240);
        } else {
          context.drawImage(image, 0, 0, canvas.width, canvas.height);
        }
        // Ensure Chromium presents fresh frames even for the same source image.
        context.fillStyle = `rgb(${Math.floor(performance.now()) % 255}, 0, 0)`;
        context.fillRect(0, 0, 1, 1);
      };
      draw();
      setInterval(draw, 1000 / 30);
      navigator.mediaDevices.getUserMedia = async () => {
        const stream = canvas.captureStream(30);
        window.cameraTest.tracks.push(...stream.getTracks());
        return stream;
      };
    }, base64);
    await page.getByRole("button", { name: "Start kamera" }).click();
    await expect(page.locator(".result-letter")).toHaveText("A", { timeout: 45000 });
    // Measure completed inference, including transfer, rather than video rendering.
    await expect
      .poll(
        () =>
          page
            .getByLabel("Tracking frame rate")
            .textContent()
            .then((text) => parseInt(text || "0")),
        { timeout: 15000 },
      )
      .toBeGreaterThan(10);

    await page.evaluate(() => {
      window.cameraTest.multiple = true;
    });
    await expect
      .poll(() =>
        page
          .locator(".result-meta")
          .textContent()
          .then((text) => parseInt(text || "0")),
      )
      .toBeGreaterThan(2);
    const still = await page.evaluate(() => {
      const video = document.querySelector("video")!;
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext("2d")!.drawImage(video, 0, 0);
      return window.desktop!.recognize(canvas.toDataURL("image/jpeg", 0.95));
    });
    expect(still.hands.length).toBeGreaterThan(2);
    for (const hand of still.hands) {
      expect(hand.landmarks).toHaveLength(21);
      expect(hand.features).toHaveLength(63);
    }

    await page.getByRole("button", { name: "Stopp kamera" }).click();
    await expect(page.getByText("Kamera av")).toBeVisible();
    expect(
      await page.locator("video").evaluate((element) => (element as HTMLVideoElement).srcObject),
    ).toBeNull();
    expect(
      await page.evaluate(() =>
        window.cameraTest.tracks.every((track) => track.readyState === "ended"),
      ),
    ).toBe(true);

    await page.evaluate(() => {
      window.cameraTest.multiple = false;
    });
    await page.getByRole("button", { name: "Start kamera" }).click();
    await expect(page.locator(".result-letter")).toHaveText("A");
    await expect(page.getByText("1 hånd funnet")).toBeVisible();
    await page.getByRole("button", { name: "Bilde", exact: true }).click();
    expect(
      await page.evaluate(() =>
        window.cameraTest.tracks.every((track) => track.readyState === "ended"),
      ),
    ).toBe(true);
    await page.getByRole("button", { name: "Live kamera", exact: true }).click();
    await expect(page.getByText("Kamera av")).toBeVisible();
    await page.getByRole("button", { name: "Start kamera" }).click();
    await expect(page.locator(".result-letter")).toHaveText("A");
    await page.getByRole("button", { name: "Stopp kamera" }).click();
  } finally {
    await application.close();
  }
});
