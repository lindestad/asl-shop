import { test, expect, _electron as electron } from "@playwright/test";
import { existsSync } from "node:fs";
import path from "node:path";

test("Electron opens an image and returns real hand detection without a fabricated letter", async () => {
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
    await expect(
      page.getByText("Hånd funnet. Bokstavmodellen er ikke koblet til ennå."),
    ).toBeVisible();
    expect(
      await page.evaluate(() => typeof (window as unknown as { require?: unknown }).require),
    ).toBe("undefined");
    expect(await page.evaluate(() => typeof window.desktop?.recognize)).toBe("function");
  } finally {
    await application.close();
  }
});
