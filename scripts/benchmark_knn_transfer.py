"""Legacy CSV KNN transfer to OpenFS weak frame labels using matching XY features."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.neural import CONTRACT, HANDSHAPES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=ROOT / 'data/extracted_landmarks.csv')
    parser.add_argument('--packed', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    schema = json.loads((args.packed / 'dataset.json').read_text())
    packed = (args.packed / schema.get('generation', '')).resolve()
    if schema['contract'] != CONTRACT or not packed.is_relative_to(args.packed.resolve()):
        raise ValueError('Incompatible packed data contract or generation path.')
    frame = pd.read_csv(args.csv)
    columns = [f'{axis}{index}' for index in range(21) for axis in 'xyz']
    if list(frame.columns) != ['label', *columns]:
        raise ValueError('Expected the legacy label + 63-coordinate CSV.')
    frame = frame[frame.label.isin(HANDSHAPES[1:])]
    xy = frame[columns].to_numpy().reshape(-1, 21, 3)[..., :2].copy()
    # Original CSV is already wrist-relative and mirrored. Replace XYZ scaling
    # with the new XY MCP9 scaling without inventing depth or image dimensions.
    xy /= np.linalg.norm(xy[:, 9], axis=-1)[:, None, None] + 1e-6
    if not np.isfinite(xy).all():
        raise ValueError('Non-finite legacy landmarks.')
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(xy.reshape(-1, 42), frame.label.to_numpy())
    with np.load(packed / 'test_handshape.npz') as archive:
        features, labels = archive['features'], archive['labels']
    started = time.perf_counter()
    correct = accepted = correct_accepted = 0
    for start in range(0, len(labels), 1024):
        scores = classifier.predict_proba(features[start:start + 1024])
        predicted = classifier.classes_[scores.argmax(-1)]
        expected = np.array(HANDSHAPES)[labels[start:start + 1024]]
        match = predicted == expected
        keep = scores.max(-1) >= .6
        correct += int(match.sum())
        accepted += int(keep.sum())
        correct_accepted += int((match & keep).sum())
    result = dict(model='KNN5 re-trained on all legacy alphabet CSV static letters using matching42XY normalization',
                  training_rows=len(frame), test_frames=len(labels), accuracy=correct / len(labels),
                  accepted_coverage=accepted / len(labels), accepted_accuracy=correct_accepted / accepted if accepted else None,
                  threshold=.6, seconds=time.perf_counter() - started,
                  label_provenance='OpenFS refined pseudo-label agreement, not human live letters',
                  comparison='Different training data and architecture; not an architecture-only ablation.')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
