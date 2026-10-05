export type Workspace = "editor" | "camera" | "batch";
export type Point = { x: number; y: number; z: number };
export type Hand = {
  hand: "Left" | "Right";
  score: number;
  features: number[];
  landmarks: Point[];
};
export type Recognition = {
  hands: Hand[];
  width: number;
  height: number;
  classifierAvailable: boolean;
  prediction: { letter: string; confidence: number } | null;
};
export type SourceImage = { name: string; dataUrl: string };
export type BatchExport = SourceImage & { recognition: Recognition | null };
export interface DesktopBridge {
  openImages: (multiple: boolean) => Promise<SourceImage[]>;
  saveImage: (image: SourceImage) => Promise<string | null>;
  saveBatch: (images: BatchExport[]) => Promise<string | null>;
  recognize: (dataUrl: string) => Promise<Recognition>;
  recognizeFrame: (dataUrl: string) => Promise<Recognition>;
}
declare global {
  interface Window {
    desktop?: DesktopBridge;
  }
}
