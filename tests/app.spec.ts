import { test, expect } from "@playwright/test";

const pixel = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a7u0AAAAASUVORK5CYII=",
  "base64",
);

test("shows only camera and image recognition views", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Live kamera" })).toBeVisible();
  await expect(page.getByRole("navigation", { name: "Visninger" }).getByRole("button")).toHaveCount(
    2,
  );

  await page.getByRole("button", { name: "Bilde" }).click();
  await expect(page.getByRole("heading", { name: "Gjenkjenn fra bilde" })).toBeVisible();
  await page.getByTestId("image-input").setInputFiles({
    name: "tegn.png",
    mimeType: "image/png",
    buffer: pixel,
  });
  await expect(page.getByRole("img", { name: "tegn.png" })).toBeVisible();
  await expect(page.getByText("Bokstavmodellen er ikke koblet til ennå.")).toHaveCount(0);
  await expect(page.getByText("Start Electron-appen for lokal gjenkjenning.")).toBeVisible();
});

test("stops the camera when leaving the live view", async ({ page, context }) => {
  await context.grantPermissions(["camera"]);
  await page.goto("/");
  await page.getByRole("button", { name: "Start kamera" }).click();
  await expect(page.getByText("Kamera på")).toBeVisible();
  await page.getByRole("button", { name: "Bilde" }).click();
  await expect(page.getByRole("heading", { name: "Gjenkjenn fra bilde" })).toBeVisible();
  await page.getByRole("button", { name: "Live kamera" }).click();
  await expect(page.getByText("Kamera av")).toBeVisible();
});

test("shows independent neural letter and sequence outputs, finishes and resets text", async ({
  page,
  context,
}) => {
  await context.grantPermissions(["camera"]);
  await page.addInitScript(() => {
    window.desktop = {
      openImage: async () => null,
      recognize: async () => {
        throw new Error("Unused image request");
      },
      recognizeFrame: async (_image, options) => ({
        hands: [],
        width: 640,
        height: 480,
        classifierAvailable: true,
        handshapeAvailable: true,
        prediction: { letter: "B", confidence: 0.95, method: "neural" },
        fingerspelling: {
          available: true,
          text: "AAB",
          frames: 12,
          status: options.finish ? "finished" : "recognizing",
          provisional: !options.finish,
        },
      }),
    };
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Start kamera" }).click();
  await expect(page.locator(".result-letter")).toHaveText("B");
  await expect(page.locator(".spelled-text")).toHaveText("AAB");
  await expect(page.getByText("Nevral bokstavmodell", { exact: false })).toBeVisible();
  await page.getByRole("button", { name: "Fullfør tekst" }).click();
  await expect(page.getByText("Sekvensen er ferdig.")).toBeVisible();
  await page.getByRole("button", { name: "Ny sekvens" }).click();
  await expect(page.getByText("Foreløpig tekst", { exact: false })).toBeVisible();
});
