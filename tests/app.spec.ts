import { test, expect } from "@playwright/test";

test("shows only the live camera view", async ({ page, context }) => {
  await context.grantPermissions(["camera"]);
  await page.goto("/");
  await expect(page).toHaveTitle("ASL alphabet recognition");
  await expect(page.getByLabel("ASL alphabet recognition")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Live camera" })).toBeVisible();
  await expect(page.getByRole("navigation")).toHaveCount(0);

  await page.getByRole("button", { name: "Start camera" }).click();
  await expect(page.getByText("Camera on")).toBeVisible();
  await page.getByRole("button", { name: "Stop camera" }).click();
  await expect(page.getByText("Camera off")).toBeVisible();
});

test("fits on one screen without scrolling", async ({ page }) => {
  await page.goto("/");
  // The default and minimum Electron window sizes.
  for (const size of [
    { width: 1280, height: 840 },
    { width: 760, height: 600 },
  ]) {
    await page.setViewportSize(size);
    await expect(page.getByRole("textbox", { name: "Text" })).toBeInViewport({ ratio: 1 });
    expect(
      await page.evaluate(() => {
        const root = document.documentElement;
        return root.scrollHeight <= root.clientHeight && root.scrollWidth <= root.clientWidth;
      }),
    ).toBe(true);
  }
});

test("types held signs into the scratchpad", async ({ page }) => {
  await page.addInitScript(() => {
    const canvas = document.createElement("canvas");
    canvas.width = 320;
    canvas.height = 240;
    const context = canvas.getContext("2d")!;
    setInterval(() => {
      context.fillStyle = `rgb(${Math.floor(performance.now()) % 255}, 0, 0)`;
      context.fillRect(0, 0, canvas.width, canvas.height);
    }, 1000 / 30);
    navigator.mediaDevices.getUserMedia = async () => canvas.captureStream(30);
    const state = { sign: "nothing" };
    (window as unknown as { sign: typeof state }).sign = state;
    window.desktop = {
      openImage: async () => null,
      recognize: async () => {
        throw new Error("unused");
      },
      recognizeFrame: async () => ({
        hands: [],
        width: 320,
        height: 240,
        model: "knn",
        classifierAvailable: true,
        prediction: { letter: state.sign, confidence: 1 },
      }),
    };
  });
  await page.goto("/");
  const text = page.getByRole("textbox", { name: "Text" });
  const show = (sign: string) =>
    page.evaluate((value) => {
      (window as unknown as { sign: { sign: string } }).sign.sign = value;
    }, sign);

  await page.getByRole("button", { name: "Start camera" }).click();
  await show("H");
  await expect(text).toHaveValue("H");
  // Holding the same sign longer does not repeat it.
  await page.waitForTimeout(1500);
  await expect(text).toHaveValue("H");
  await show("I");
  await expect(text).toHaveValue("HI");
  await show("space");
  await expect(text).toHaveValue("HI ");
  await show("del");
  await expect(text).toHaveValue("HI");

  await page.getByRole("button", { name: "Backspace" }).click();
  await expect(text).toHaveValue("H");
  await page.getByRole("button", { name: "Space", exact: true }).click();
  await expect(text).toHaveValue("H ");
  await page.getByRole("button", { name: "Clear" }).click();
  await expect(text).toHaveValue("");
});

test("sends frames to the model chosen in the dropdown", async ({ page }) => {
  await page.addInitScript(() => {
    const canvas = document.createElement("canvas");
    canvas.width = 320;
    canvas.height = 240;
    const context = canvas.getContext("2d")!;
    setInterval(() => {
      context.fillStyle = `rgb(${Math.floor(performance.now()) % 255}, 0, 0)`;
      context.fillRect(0, 0, canvas.width, canvas.height);
    }, 1000 / 30);
    navigator.mediaDevices.getUserMedia = async () => canvas.captureStream(30);
    window.desktop = {
      openImage: async () => null,
      recognize: async () => {
        throw new Error("unused");
      },
      recognizeFrame: async (_, model = "knn") => ({
        hands: [],
        width: 320,
        height: 240,
        model,
        classifierAvailable: true,
        prediction:
          model === "cnn" ? { letter: "B", confidence: 0.87 } : { letter: "A", confidence: 1 },
      }),
    };
  });
  await page.goto("/");
  const dropdown = page.getByLabel("Model");
  await expect(dropdown).toHaveValue("knn");

  await page.getByRole("button", { name: "Start camera" }).click();
  await expect(page.locator(".result-letter")).toHaveText("A");
  await expect(page.getByText("Closest match in the training examples")).toBeVisible();

  await dropdown.selectOption("cnn");
  await expect(page.locator(".result-letter")).toHaveText("B");
  await expect(page.getByText("87% confidence")).toBeVisible();

  await dropdown.selectOption("knn");
  await expect(page.locator(".result-letter")).toHaveText("A");
});
