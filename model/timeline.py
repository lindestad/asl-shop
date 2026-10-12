"""Pure timestamp resampling for the nominal-30-Hz sequence-model contract."""
import numpy as np


class TimelineResampler:
    def __init__(self, fps=30, max_interpolation_gap_ms=250):
        if not np.isfinite(fps) or not np.isfinite(max_interpolation_gap_ms) or fps <= 0 or max_interpolation_gap_ms <= 0:
            raise ValueError("Positive frame rate and interpolation gap are required.")
        self.period_ms = 1000 / fps
        self.max_gap_ms = max_interpolation_gap_ms
        self.previous = None
        self.previous_time = None
        self.origin = None
        self.next_index = 0

    def feed(self, frame, timestamp_ms):
        frame = np.asarray(frame, dtype=np.float32)
        if frame.shape != (91,) or not np.isfinite(frame).all() or not np.isfinite(timestamp_ms):
            raise ValueError("Expected finite 91-value features and a finite timestamp.")
        if self.previous_time is not None and timestamp_ms <= self.previous_time:
            raise ValueError("Resampler capture timestamps must strictly increase.")
        if self.previous is None:
            value = frame.copy()
            value[-1] = 0
            for base in (0, 45):
                if value[base + 44] <= 0:
                    value[base:base + 45] = 0
            self.previous, self.previous_time = frame.copy(), timestamp_ms
            self.origin, self.next_index = timestamp_ms, 1
            return [(timestamp_ms, value)]
        generated = []
        interval = timestamp_ms - self.previous_time
        while True:
            tick = self.origin + self.next_index * self.period_ms
            if tick > timestamp_ms + 1e-6:
                break
            value = np.zeros(91, dtype=np.float32)
            value[-1] = self.period_ms / 1000
            at_endpoint = abs(tick - timestamp_ms) <= 1e-6
            for base in (0, 45):
                if at_endpoint:
                    if frame[base + 44] > 0:
                        value[base:base + 45] = frame[base:base + 45]
                elif interval <= self.max_gap_ms and self.previous[base + 44] > 0 and frame[base + 44] > 0:
                    weight = (tick - self.previous_time) / interval
                    value[base:base + 44] = self.previous[base:base + 44] * (1 - weight) + frame[base:base + 44] * weight
                    value[base + 44] = 1
                # Presence transitions and long acquisition gaps stay missing;
                # they never synthesize an unobserved hand or carry its pose forward.
            generated.append((tick, value))
            self.next_index += 1
        self.previous, self.previous_time = frame.copy(), timestamp_ms
        return generated


def resample_frames(frames, timestamps_ms, *, fps=30, max_interpolation_gap_ms=250):
    """Offline replay uses exactly the same grid and masks as live acquisition."""
    if len(frames) != len(timestamps_ms):
        raise ValueError("Frame and capture timestamp counts differ.")
    resampler = TimelineResampler(fps, max_interpolation_gap_ms)
    outputs = [item for frame, timestamp in zip(frames, timestamps_ms) for item in resampler.feed(frame, timestamp)]
    if not outputs:
        return np.empty((0, 91), dtype=np.float32), np.empty(0, dtype=np.float64)
    times, values = zip(*outputs)
    return np.stack(values), np.array(times, dtype=np.float64)
