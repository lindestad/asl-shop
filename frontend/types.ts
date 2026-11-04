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
    method?: "causal-tcn" | "openfs" | "ensemble";
    refinementError?: string;
    contextStatus?: "collecting" | "spelling" | "held-letter" | "insufficient-context";
    contextMessage?: string;
    refiner?: {
      enabled: boolean;
      available: boolean;
      status: "disabled" | "configured" | "ready" | "error";
      error?: string;
    };
    ensemble?: {
      enabled: boolean;
      available: boolean;
      members?: number;
      status: "disabled" | "configured" | "ready" | "error";
      error?: string;
    };
  };
};

export type ReplayHand = { hand: "Right" | "Left"; points: [number, number][] };
export type ReplayFrame = { t: number; hands: ReplayHand[] };
export type DemoClipInfo = {
  index: number;
  name: string;
  signer: string;
  frames: number;
  durationMs: number;
  text: string;
};
export type DemoClip = DemoClipInfo & { frames: ReplayFrame[] };

export type SourceImage = { name: string; dataUrl: string };

export interface DesktopBridge {
  openImage: () => Promise<SourceImage | null>;
  recognize: (dataUrl: string) => Promise<Recognition>;
  recognizeFrame: (
    dataUrl: string,
    options: { sessionId: string; timestampMs: number; finish?: boolean },
  ) => Promise<Recognition>;
  recognizeLandmarks: (
    landmarks: ReplayHand[],
    options: { sessionId: string; timestampMs: number; finish?: boolean },
  ) => Promise<Recognition>;
  demoList: () => Promise<DemoClipInfo[]>;
  demoClip: (index: number) => Promise<DemoClip>;
}

declare global {
  interface Window {
    desktop?: DesktopBridge;
  }
}
