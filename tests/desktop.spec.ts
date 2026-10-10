import { test, expect, _electron as electron } from "@playwright/test";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import path from "node:path";

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
    await application.evaluate(({ dialog }, filename) => {
      dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [filename] });
    }, sample);

    await page.getByRole("button", { name: "Bilde" }).click();
    await page.getByRole("button", { name: "Velg bilde" }).click();
    await expect(page.getByRole("img", { name: "A1000.jpg" })).toBeVisible();
    await expect(page.getByText("1 hånd funnet")).toBeVisible({ timeout: 45000 });
    await expect(page.locator(".result-letter")).toHaveText("A");
    await expect(page.getByText("5 av 5 nærmeste treningseksempler")).toBeVisible();
    expect(
      await page.evaluate(() => typeof (window as unknown as { require?: unknown }).require),
    ).toBe("undefined");
    expect(await page.evaluate(() => typeof window.desktop?.recognize)).toBe("function");
  } finally {
    await application.close();
  }
});

test("live camera frames use the trained letter model", async () => {
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
      canvas.width = 640;
      canvas.height = 480;
      const context = canvas.getContext("2d")!;
      const draw = () => {
        context.drawImage(image, 0, 0, canvas.width, canvas.height);
        context.fillStyle = `rgb(${Math.floor(performance.now()) % 255}, 0, 0)`;
        context.fillRect(0, 0, 1, 1);
      };
      draw();
      setInterval(draw, 1000 / 15);
      navigator.mediaDevices.getUserMedia = async () => canvas.captureStream(15);
    }, base64);
    await page.getByRole("button", { name: "Start kamera" }).click();
    await expect(page.locator(".result-letter")).toHaveText("A", { timeout: 45000 });
    await page.getByRole("button", { name: "Stopp kamera" }).click();
    await expect(page.getByText("Kamera av")).toBeVisible();
  } finally {
    await application.close();
  }
});
