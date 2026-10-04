"""How much of the ground-truth pose motion is frame-to-frame high-frequency content?

`motion_ratio` (pose_metrics) is generated path length / reference path length. A
deterministic regressor cannot predict frame-level jitter in the reference (tracker noise,
fine finger movement), so its motion ratio is capped below 1 by however much of the
reference's path length lives in that jitter. This measures that directly: smooth the
reference with a moving average and see how much path length survives.

    python scripts/diagnostics/gt_motion_smoothing.py <TENSOR_DIR> [signer_token]

TENSOR_DIR holds <sentence_id>/*.npy; signer_token defaults to _U4_. Needs numpy only.
"""
import glob
import sys

import numpy as np

tensor_dir = sys.argv[1]
token = sys.argv[2] if len(sys.argv) > 2 else "_U4_"
files = sorted(glob.glob(f"{tensor_dir}/*/*{token}*.npy"))
if not files:
    raise SystemExit(f"no files matching {token} under {tensor_dir}")


def pathlen(p):
    return np.linalg.norm(np.diff(p, axis=0), axis=-1).mean(axis=-1).sum()


def accel(p):
    return np.linalg.norm(np.diff(p, n=2, axis=0), axis=-1).mean()


def smooth(p, k):
    pad = k // 2
    q = np.pad(p, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    c = np.concatenate([np.zeros((1,) + q.shape[1:]), np.cumsum(q, axis=0)])
    return (c[k:] - c[:-k]) / k


windows = (3, 5, 7, 15)
kept = {k: [] for k in windows}
acc = {k: [] for k in windows}
for f in files:
    p = np.load(f).astype(np.float64)
    if p.ndim == 2:
        p = p.reshape(len(p), 95, 3)
    l0, a0 = pathlen(p), accel(p)
    for k in windows:
        s = smooth(p, k)
        kept[k].append(pathlen(s) / l0)
        acc[k].append(accel(s) / a0)

print(f"{len(files)} recordings ({token}); ground-truth statistics after moving-average smoothing")
print("(1.0 = unchanged)")
for k in windows:
    print(f"  window {k:2d}: path length x{np.mean(kept[k]):.3f}   acceleration x{np.mean(acc[k]):.3f}")
