import type { SourceImage } from "./types";

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
  [13, 17],
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20],
];

export async function readImage(file: File): Promise<SourceImage> {
  if (!["image/png", "image/jpeg", "image/webp"].includes(file.type))
    throw new Error("Velg et PNG-, JPEG- eller WebP-bilde.");
  if (file.size > 18 * 1024 * 1024) throw new Error("Bildet må være mindre enn 18 MB.");
  const dataUrl = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error("Kunne ikke lese bildet."));
    reader.readAsDataURL(file);
  });
  return { name: file.name, dataUrl };
}
