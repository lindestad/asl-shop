import type { Hand, SourceImage } from "./types";

export const CONNECTIONS = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20],
];

export async function loadImage(dataUrl: string): Promise<HTMLImageElement> {
  const image = new Image();
  image.src = dataUrl;
  await image.decode();
  return image;
}

export async function readFiles(files: File[]): Promise<SourceImage[]> {
  if (files.length > 100) throw new Error("Choose up to 100 images at a time.");
  return Promise.all(
    files.map(async (file) => {
      if (!["image/png", "image/jpeg", "image/webp"].includes(file.type))
        throw new Error(`${file.name}: choose a PNG, JPEG, or WebP image.`);
      if (file.size > 30 * 1024 * 1024) throw new Error(`${file.name} is larger than 30 MB.`);
      return {
        name: file.name,
        dataUrl: await new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result as string);
          reader.onerror = () => reject(new Error(`Could not read ${file.name}.`));
          reader.readAsDataURL(file);
        }),
      };
    }),
  );
}

export function downloadImage(image: SourceImage) {
  const link = document.createElement("a");
  link.href = image.dataUrl;
  link.download = image.name;
  link.click();
}

export async function saveImage(image: SourceImage): Promise<boolean> {
  if (window.desktop) return (await window.desktop.saveImage(image)) !== null;
  downloadImage(image);
  return true;
}

export function drawLandmarks(
  context: CanvasRenderingContext2D,
  hands: Hand[],
  width: number,
  height: number,
) {
  context.save();
  context.lineWidth = Math.max(1.5, width / 300);
  context.strokeStyle = "#fff";
  context.fillStyle = "#171717";
  context.shadowColor = "#000";
  context.shadowBlur = 3;
  for (const hand of hands) {
    for (const [a, b] of CONNECTIONS) {
      context.beginPath();
      context.moveTo(hand.landmarks[a].x * width, hand.landmarks[a].y * height);
      context.lineTo(hand.landmarks[b].x * width, hand.landmarks[b].y * height);
      context.stroke();
    }
    for (const point of hand.landmarks) {
      context.beginPath();
      context.arc(point.x * width, point.y * height, Math.max(3, width / 150), 0, Math.PI * 2);
      context.fill();
      context.stroke();
    }
  }
  context.restore();
}

export type Filter = "grayscale" | "blur" | "sobel" | "binary";
export function applyFilter(canvas: HTMLCanvasElement, filter: Filter, threshold: number) {
  const context = canvas.getContext("2d", { willReadFrequently: true })!;
  if (filter === "blur") {
    const copy = document.createElement("canvas");
    copy.width = canvas.width;
    copy.height = canvas.height;
    const ctx = copy.getContext("2d")!;
    ctx.filter = "blur(3px)";
    ctx.drawImage(canvas, 0, 0);
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(copy, 0, 0);
    return;
  }
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height);
  const gray = new Uint8Array(canvas.width * canvas.height);
  for (let index = 0; index < gray.length; index++) {
    const p = index * 4;
    gray[index] = Math.round(
      pixels.data[p] * 0.299 + pixels.data[p + 1] * 0.587 + pixels.data[p + 2] * 0.114,
    );
  }
  for (let y = 0; y < canvas.height; y++) {
    for (let x = 0; x < canvas.width; x++) {
      const index = y * canvas.width + x;
      let value = gray[index];
      if (filter === "binary") value = value >= threshold ? 255 : 0;
      if (filter === "sobel") {
        const at = (dx: number, dy: number) =>
          gray[
            Math.max(0, Math.min(canvas.height - 1, y + dy)) * canvas.width +
              Math.max(0, Math.min(canvas.width - 1, x + dx))
          ];
        const gx = -at(-1, -1) + at(1, -1) - 2 * at(-1, 0) + 2 * at(1, 0) - at(-1, 1) + at(1, 1);
        const gy = -at(-1, -1) - 2 * at(0, -1) - at(1, -1) + at(-1, 1) + 2 * at(0, 1) + at(1, 1);
        value = Math.min(255, Math.hypot(gx, gy));
      }
      pixels.data[index * 4] = pixels.data[index * 4 + 1] = pixels.data[index * 4 + 2] = value;
    }
  }
  context.putImageData(pixels, 0, 0);
}
