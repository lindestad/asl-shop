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
  prediction: { letter: string; confidence: number; method?: "knn" | "neural" } | null;
  handshapeAvailable?: boolean;
  handshapeError?: string | null;
  fingerspelling?: {
    available: boolean;
    text: string;
    status: "unavailable" | "recognizing" | "finished" | "limit";
    provisional: boolean;
    frames: number;
    modelFrames?: number;
    error?: string;
    method?: "causal-tcn" | "fsboard-stream" | "openfs";
    refinementError?: string;
    contextStatus?: "collecting" | "spelling" | "held-letter" | "insufficient-context";
    contextMessage?: string;
    refiner?: {
      enabled: boolean;
      available: boolean;
      status: "disabled" | "configured" | "ready" | "error";
      error?: string;
    };
  };
};

export type SourceImage = { name: string; dataUrl: string };

export interface DesktopBridge {
  openImage: () => Promise<SourceImage | null>;
  recognize: (dataUrl: string) => Promise<Recognition>;
  recognizeFrame: (
    dataUrl: string,
    options: { sessionId: string; timestampMs: number; finish?: boolean },
  ) => Promise<Recognition>;
}

declare global {
  interface Window {
    desktop?: DesktopBridge;
  }
}
