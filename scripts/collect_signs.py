"""Explicit local landmark recording; press SPACE to begin, SPACE again to save, Q to cancel."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import cv2
    from handTracker.hand_features import detector, extractHands
    from model.neural import frame_features
    from model.sessions import write_clip
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "sessions")
    parser.add_argument("--signer-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--kind", required=True, choices=["held", "spelling", "transition", "idle"])
    parser.add_argument("--transcript", default="")
    parser.add_argument("--hand", choices=["Left", "Right"])
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    frames, times, recording = [], [], False
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError("Could not open camera.")
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                raise RuntimeError("Camera stopped.")
            now = time.monotonic_ns() / 1e6
            hands = extractHands(image, detector)
            if recording:
                frames.append(frame_features(hands, image.shape[1], image.shape[0],
                                             now - times[-1] if times else 0))
                times.append(now)
            preview = image.copy()
            cv2.putText(preview, "Recording landmarks" if recording else "SPACE: record | Q: cancel",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2)
            cv2.imshow("ASL collection", preview)
            key = cv2.waitKey(1) & 0xff
            if key in (ord("q"), 27):
                return
            if key == 32:
                if not recording:
                    recording = True
                else:
                    entry = write_clip(args.output, frames, times, signer_id=args.signer_id,
                                       session_id=args.session_id, kind=args.kind, transcript=args.transcript,
                                       hand=args.hand, split=args.split, source=f"webcam:{args.camera}",
                                       label_provenance="human-clip-label")
                    print(entry["clip_id"])
                    return
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
