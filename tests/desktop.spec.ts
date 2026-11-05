import { test, expect, _electron as electron } from "@playwright/test";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import path from "node:path";

declare global {
  interface Window {
    cameraTest: {
      multiple: boolean;
      tracks: MediaStreamTrack[];
    };
  }
}

test.beforeEach(() => {
  const sample = path.resolve("data/asl_alphabet_train/asl_alphabet_train/A/A1000.jpg");
  test.skip(!existsSync(sample), "Download the ASL dataset to run the desktop integration tests.");
  if (!process.env.ASL_SHOP_TEST_APP) {
    expect(
      existsSync(path.resolve("data/knn.pkl")),
      "Train the classifier with model/knn.py first.",
    ).toBe(true);
  }
});

test("Electron recognizes a letter from an image with the trained model", async () => {
  const sample = path.resolve("data/asl_alphabet_train/asl_alphabet_train/A/A1000.jpg");
  test.skip(!existsSync(sample), "Download the ASL dataset to run the desktop integration test.");
  const application = await electron.launch({
    executablePath: process.env.ASL_SHOP_TEST_APP,
    args: process.env.ASL_SHOP_TEST_APP ? [] : ["."],
    env: { ...process.env, ASL_DEV_URL: "http://127.0.0.1:5173" },
  });
  try {
    const page = await application.firstWindow();
    await page.waitForLoadState("domcontentloaded");
    const dataUrl = `data:image/jpeg;base64,${(await readFile(sample)).toString("base64")}`;
    const result = await page.evaluate((url) => window.desktop!.recognize(url), dataUrl);
    expect(result.hands).toHaveLength(1);
    expect(result.prediction?.letter).toBe("A");
    expect(result.prediction?.confidence).toBe(1);
    expect(
      await page.evaluate(() => typeof (window as unknown as { require?: unknown }).require),
    ).toBe("undefined");
    expect(await page.evaluate(() => typeof window.desktop?.recognize)).toBe("function");
  } finally {
    await application.close();
  }
});

test("camera recognizes letters, keeps up with video, finds new hands, and restarts", async () => {
  const sample = path.resolve("data/asl_alphabet_train/asl_alphabet_train/A/A1000.jpg");
  test.skip(!existsSync(sample), "Download the ASL dataset to run the desktop integration test.");
  const application = await electron.launch({
    executablePath: process.env.ASL_SHOP_TEST_APP,
    args: process.env.ASL_SHOP_TEST_APP ? [] : ["."],
    env: { ...process.env, ASL_DEV_URL: "http://127.0.0.1:5173" },
  });
  try {
    const page = await application.firstWindow();
    await page.waitForLoadState("domcontentloaded");
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
    await page.getByRole("button", { name: "Start camera" }).click();
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

    await page.getByRole("button", { name: "Stop camera" }).click();
    await expect(page.getByText("Camera off")).toBeVisible();
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
    await page.getByRole("button", { name: "Start camera" }).click();
    await expect(page.locator(".result-letter")).toHaveText("A");
    await expect(page.getByText("1 hand found")).toBeVisible();
    await page.getByRole("button", { name: "Stop camera" }).click();
  } finally {
    await application.close();
  }
});

test("demo replay runs dataset footage through the live path", async () => {
  // Model load plus a real-time clip replay does not fit the default 60 s.
  test.setTimeout(150000);
  const footage = path.resolve(
    "data/external/openfs/demo-clips/67fc_535_20221021_200720_909133_911200.mp4",
  );
  const packedReady =
    existsSync(path.resolve("data/openfs-packed/test_clips.json")) &&
    existsSync(path.resolve("data/openfs-packed/test_frames.npy"));
  test.skip(
    !packedReady || !existsSync(footage),
    "Pack the OpenFS test split and fetch demo footage to run the replay test.",
  );
  const application = await electron.launch({
    executablePath: process.env.ASL_SHOP_TEST_APP,
    args: process.env.ASL_SHOP_TEST_APP ? [] : ["."],
    env: { ...process.env, ASL_DEV_URL: "http://127.0.0.1:5173" },
  });
  try {
    const page = await application.firstWindow();
    await page.waitForLoadState("domcontentloaded");
    const card = page.getByRole("region", { name: "Dataset replay" });
    await expect(card).toBeVisible({ timeout: 30000 });
    // All four clips have footage, so scope to one option rather than a unique match.
    await expect(card.getByRole("option", { name: /footage/ }).first()).toBeAttached();
    await card.getByRole("button", { name: "Play clip" }).click();
    await expect(card.getByRole("button", { name: "Replaying…" })).toBeVisible();
    // A tainted canvas or a dead protocol rejects every exported frame, so the
    // replay only re-enables and only finishes when detection really ran.
    await expect(card.getByRole("button", { name: "Play clip" })).toBeEnabled({ timeout: 45000 });
    await expect(page.locator("p.error")).toBeHidden();
    await expect(page.getByText("Sekvensen er ferdig.")).toBeVisible({ timeout: 20000 });
  } finally {
    await application.close();
  }
});
