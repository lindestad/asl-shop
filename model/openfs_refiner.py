"""Optional research-only final-text adapter for an explicitly trusted OpenFS checkout.

The upstream source and checkpoint are external dependencies, never bundled here.
"""
import importlib
import os
from pathlib import Path
import sys

import numpy as np


def preprocess(right, left):
    """Match released per-axis box normalization and compressed detection indices."""
    import torch
    poses, segments, indices = [], [], []
    for position, person in enumerate(sorted(set(right) | set(left))):
        for side, stream in enumerate((right, left)):
            xy = np.asarray(stream.get(person, []), dtype=np.float32)
            if not len(xy):
                continue
            if xy.ndim != 3 or xy.shape[1] != 21 or xy.shape[2] not in (2, 3) or not np.isfinite(xy).all():
                raise ValueError("OpenFS expects finite detection-ordered T × 21 × XY/XYZ poses.")
            xy = xy[..., :2].copy()
            xy -= (xy.min(axis=1, keepdims=True) + xy.max(axis=1, keepdims=True)) * .5
            xy /= np.abs(xy).max(axis=1, keepdims=True) + 1e-6
            xy *= .5
            poses.append(torch.from_numpy(xy))
            segments.append(torch.full((len(xy),), 2 * position + side, dtype=torch.long))
            indices.append(torch.arange(len(xy), dtype=torch.long))
    if not poses:
        return None
    return torch.cat(poses), torch.cat(segments), torch.cat(indices)


def load_model(source, model_directory):
    """Use the fixed upstream constructor, not an executable target from YAML."""
    import torch
    import yaml
    source = Path(source).resolve()
    directory = Path(model_directory).resolve()
    expected = source / "lib/models/recognizer.py"
    if not expected.is_file():
        raise ValueError("OpenFS source checkout is missing lib/models/recognizer.py.")
    sys.path.insert(0, str(source))
    module = importlib.import_module("lib.models.recognizer")
    if Path(module.__file__).resolve() != expected.resolve():
        raise ValueError("Another OpenFS/lib module is already loaded; restart with the intended source checkout.")
    config = yaml.safe_load((directory / ".hydra/config.yaml").read_text(encoding="utf-8"))
    chars = config["decode"]["chars"]
    if not isinstance(chars, str) or not chars or len(set(chars)) != len(chars):
        raise ValueError("OpenFS checkpoint vocabulary is invalid.")
    kwargs = {key: value for key, value in config["recognizer"].items()
              if not key.startswith("_") and key != "ckpt_path"}
    network = module.EncoderDecoderTransformer(char_size=len(chars) + 1, **kwargs).cpu().eval()
    network.load_state_dict(torch.load(directory / "best_model.pth", map_location="cpu", weights_only=True), strict=True)
    return network, chars


def decode_output(tokens, chars):
    result = []
    for token in tokens[1:]:
        token = int(token)
        if token in (0, len(chars) + 1):
            break
        if not 0 <= token < len(chars):
            raise ValueError("OpenFS emitted an invalid special token.")
        result.append(chars[token])
    else:
        raise ValueError("OpenFS output reached the decoding limit; finish a shorter segment.")
    return "".join(result).upper()


class OpenFSRefiner:
    def __init__(self, source=None, model_directory=None):
        self.source = source or os.environ.get("ASL_OPENFS_SOURCE")
        self.directory = model_directory or os.environ.get("ASL_OPENFS_MODEL_DIR")
        self.enabled = bool(self.source or self.directory)
        self.network = None
        self.chars = None
        self.error = None

    def readiness(self):
        if not self.enabled:
            return {"enabled": False, "status": "disabled", "available": False}
        error = self.error
        if not self.source or not self.directory:
            error = "Set both ASL_OPENFS_SOURCE and ASL_OPENFS_MODEL_DIR for optional final-text refinement."
        elif not (Path(self.source) / "lib/models/recognizer.py").is_file() or not (Path(self.directory) / "best_model.pth").is_file() or not (Path(self.directory) / ".hydra/config.yaml").is_file():
            error = "OpenFS source, checkpoint or .hydra/config.yaml is missing."
        return {"enabled": True, "available": self.network is not None and error is None,
                "status": "error" if error else "ready" if self.network is not None else "configured",
                **({"error": error} if error else {})}

    def refine(self, right, left):
        import torch
        prepared = preprocess(right, left)
        if prepared is None:
            return ""
        status = self.readiness()
        if status.get("error"):
            raise ValueError(status["error"])
        try:
            if self.network is None:
                self.network, self.chars = load_model(self.source, self.directory)
            poses, segments, indices = prepared
            if len(poses) > 1800 or int(indices.max()) >= 900:
                raise ValueError("OpenFS refinement is limited to the bounded captured segment.")
            with torch.inference_mode():
                generated = self.network.generate(poses.unsqueeze(0), segments.unsqueeze(0), indices.unsqueeze(0),
                                                  max_len=128, bos_token_id=len(self.chars), eos_token_id=0,
                                                  pad_token_id=len(self.chars) + 1)
            return decode_output(generated["output_ids"][0].tolist(), self.chars)
        except Exception as error:
            self.error = str(error)
            raise
