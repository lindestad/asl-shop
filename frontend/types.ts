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

export interface DesktopBridge {
  openImage: () => Promise<SourceImage | null>;
  recognize: (dataUrl: string) => Promise<Recognition>;
  recognizeFrame: (dataUrl: string) => Promise<Recognition>;
}

declare global {
  interface Window {
    desktop?: DesktopBridge;
  }
}
