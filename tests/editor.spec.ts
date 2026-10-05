import { test, expect } from "@playwright/test";
import type { Page } from "@playwright/test";

async function openFixture(page: Page, name = "test-image.png") {
  const data = await page.evaluate(() => {
    const canvas = document.createElement("canvas");
    canvas.width = 320;
    canvas.height = 180;
    const context = canvas.getContext("2d")!;
    context.fillStyle = "#cc4422";
    context.fillRect(0, 0, 160, 180);
    context.fillStyle = "#3388aa";
    context.fillRect(160, 0, 160, 180);
    return canvas.toDataURL().split(",")[1];
  });
  await page
    .getByTestId("image-input")
    .setInputFiles({ name, mimeType: "image/png", buffer: Buffer.from(data, "base64") });
  await expect(page.getByTestId("editor-canvas")).toHaveAttribute("width", "320");
}

test("opens an image, transforms pixels, undoes and exports a PNG", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Drop an image here" })).toBeVisible();
  await openFixture(page);
  const canvas = page.getByTestId("editor-canvas");
  await page.getByRole("button", { name: "Rotate right", exact: true }).click();
  await expect(canvas).toHaveAttribute("width", "180");
  await expect(canvas).toHaveAttribute("height", "320");
  await page.getByRole("button", { name: "Undo", exact: true }).click();
  await expect(canvas).toHaveAttribute("width", "320");
  await page.getByRole("button", { name: "Grayscale", exact: true }).click();
  await expect
    .poll(async () =>
      canvas.evaluate((element) => {
        const [r, g, b] = (element as HTMLCanvasElement)
          .getContext("2d")!
          .getImageData(20, 20, 1, 1).data;
        return r === g && g === b;
      }),
    )
    .toBe(true);
  await page.keyboard.press("Control+z");
  await expect
    .poll(async () =>
      canvas.evaluate(
        (element) =>
          (element as HTMLCanvasElement).getContext("2d")!.getImageData(20, 20, 1, 1).data[0],
      ),
    )
    .toBe(204);
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export", exact: true }).click();
  expect((await downloadPromise).suggestedFilename()).toBe("test-image.png");
  expect(errors).toEqual([]);
});

test("draws on a new canvas and crops a rectangular selection", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "New canvas", exact: true }).click();
  await page.getByRole("spinbutton", { name: "New canvas width" }).fill("320");
  await page.getByRole("spinbutton", { name: "New canvas height" }).fill("180");
  await page.getByRole("button", { name: "Create canvas", exact: true }).click();
  const canvas = page.getByTestId("editor-canvas");
  await expect(canvas).toHaveAttribute("width", "320");
  await page.getByRole("button", { name: "Brush", exact: true }).click();
  const bounds = (await canvas.boundingBox())!;
  await page.mouse.move(bounds.x + 20, bounds.y + 20);
  await page.mouse.down();
  await page.mouse.move(bounds.x + 120, bounds.y + 50, { steps: 8 });
  await page.mouse.up();
  await expect(page.getByRole("button", { name: "Undo", exact: true })).toBeEnabled();
  await expect
    .poll(async () =>
      canvas.evaluate(
        (element) =>
          (element as HTMLCanvasElement).getContext("2d")!.getImageData(20, 20, 1, 1).data[0],
      ),
    )
    .toBe(23);
  await page.getByRole("button", { name: "Rectangular selection", exact: true }).click();
  await page.mouse.move(bounds.x + 40, bounds.y + 30);
  await page.mouse.down();
  await page.mouse.move(bounds.x + 200, bounds.y + 130, { steps: 4 });
  await page.mouse.up();
  await page.getByRole("button", { name: "Crop selection", exact: true }).click();
  await expect(canvas).toHaveAttribute("width", "160");
  await expect(canvas).toHaveAttribute("height", "100");
});

test("captures a camera frame and stops its stream when returning to the editor", async ({
  page,
  context,
}) => {
  await context.grantPermissions(["camera"]);
  await page.goto("/");
  await page.getByRole("button", { name: "Live camera", exact: true }).click();
  await page.getByRole("button", { name: "Start camera", exact: true }).first().click();
  await expect(page.getByRole("button", { name: "Capture frame" })).toBeEnabled();
  await page.getByRole("button", { name: "Capture frame" }).click();
  await expect(page.getByTestId("editor-canvas")).toBeVisible();
  expect(
    await page.locator("video").evaluate((element) => (element as HTMLVideoElement).srcObject),
  ).toBeNull();
});

test("batch imports multiple images without inventing letter predictions", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Batch studio", exact: true }).click();
  await page.getByRole("button", { name: "Add images", exact: true }).last().click();
  const data = Buffer.from(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a7u0AAAAASUVORK5CYII=",
    "base64",
  );
  await page.getByTestId("image-input").setInputFiles([
    { name: "first.png", mimeType: "image/png", buffer: data },
    { name: "second.png", mimeType: "image/png", buffer: data },
  ]);
  await expect(page.locator(".batch-card")).toHaveCount(2);
  await expect(
    page.getByRole("button", { name: "Analyze all images", exact: true }),
  ).toBeDisabled();
  await page.getByRole("button", { name: "Image editor", exact: true }).click();
  await page.getByRole("button", { name: "Recognize", exact: true }).click();
  await expect(
    page.locator(".editor-workspace").getByText("Letter classifier unavailable", { exact: true }),
  ).toBeVisible();
});
