"""Render a side-by-side skeleton video for ONE held-out sentence: the real recording (left)
and the NAT+MSE model's generated signing (right).

It re-uses the generated notebook's own code (config, dataset/split, text encoder, duration
predictor, length regulator, NAT head, generation function, metrics), so the sentence is
held out under exactly the split used for evaluation and the model is exactly the evaluated
checkpoint. The sentence is picked by a fixed random seed, not by how it looks; the script
also scores a random sample of held-out sentences so you can say how typical the chosen one is.

    python scripts/make_sign_video.py --tensor-dir <.../BdSL_Pose_Tensors> --excel <xlsx> \
        --ckpt <g2p_nat_mse_best.pt> [--out-dir assets] [--pick-seed 0]

Needs: torch, transformers, tslearn, pandas, openpyxl, tqdm, opencv-python, imageio,
imageio-ffmpeg, pillow, numpy. BanglaBERT is downloaded on first use (internet).
"""
import argparse
import glob
import json
import os
import random
import sys
import tempfile

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--notebook", default="notebooks/01_full_pipeline.ipynb")
ap.add_argument("--tensor-dir", required=True)
ap.add_argument("--excel", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out-dir", default="assets")
ap.add_argument("--pick-seed", type=int, default=0)
ap.add_argument("--context-n", type=int, default=40, help="held-out sentences scored for context")
ap.add_argument("--fps", type=int, default=25)
ap.add_argument("--caption-png", default=None,
                help="pre-rendered caption image (Bengali needs complex-script shaping; Pillow's may lack it)")
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

# ---------------------------------------------------------------- reuse the notebook's code
nb = json.load(open(args.notebook, encoding="utf-8"))
cells = nb["cells"]
WANT = ("# **2. Configuration", "# **3. Dataset", "# **5. Text Encoder", "# **6. Temporal",
        "# **7. Duration", "# **9b.", "# **11. Metrics", "# **13. Inference")
selected = []
for i, c in enumerate(cells[:-1]):
    if c["cell_type"] == "markdown" and "".join(c["source"]).startswith(WANT):
        nxt = cells[i + 1]
        assert nxt["cell_type"] == "code"
        selected.append("".join(nxt["source"]))
assert len(selected) == len(WANT), (len(selected), len(WANT))

tmp = tempfile.mkdtemp(prefix="signvideo_")
OVR = f"""
cfg.EXCEL_PATH = {args.excel!r}
cfg.TENSOR_DIR = {args.tensor_dir!r}
cfg.CHECKPOINT_DIR = {tmp!r}
cfg.NUM_WORKERS = 0
cfg.CROSS_SIGNER_EVAL = False
cfg.RESUME_SOURCE_DIR = None
"""
ns = {"__name__": "__signvideo__"}
for src in selected:
    if "cfg = Config()" in src:
        src += "\n" + OVR
    exec(compile(src, "<notebook cell>", "exec"), ns)

torch, cfg, device = ns["torch"], ns["cfg"], ns["device"]

# ---------------------------------------------------------------- load the evaluated model
ns["text_encoder_nat_mse"] = ns["GlossTextEncoder"]().to(device)
ns["duration_predictor_nat_mse"] = ns["DurationPredictor"]().to(device)
ns["nat_mse_head"] = ns["NATRegressionHead"]().to(device)
ns["length_regulator"] = ns["LengthRegulator"]().to(device)
ns["temporal_pos_encoder"] = ns["TemporalPositionalEncoding"](cfg.D_MODEL).to(device)
ck = torch.load(args.ckpt, map_location=device, weights_only=False)
ns["text_encoder_nat_mse"].load_state_dict(ck["text_encoder_state_dict"])
ns["duration_predictor_nat_mse"].load_state_dict(ck["duration_state_dict"])
ns["nat_mse_head"].load_state_dict(ck["head_state_dict"])
print(f"loaded NAT+MSE checkpoint from epoch {int(ck['epoch']) + 1}")

test_ds = ns["test_ds"]
print(f"held-out test set: {len(test_ds)} sentences (single signer)")


def generate(i):
    ii, am, pose, sid, _, _ = test_ds[i]
    pred, plen = ns["generate_poses_nat_mse"](ii[None].to(device), am[None].to(device),
                                              sid[None].to(device))
    return pred[0, :int(plen[0])].float().cpu().numpy(), pose.numpy()


# ---------------------------------------------------------------- pick the sentence, score context
pick = random.Random(args.pick_seed).randrange(len(test_ds))
sample = random.Random(1).sample(range(len(test_ds)), min(args.context_n, len(test_ds)))
if pick not in sample:
    sample.append(pick)
scores = {}
for i in sample:
    p, g = generate(i)
    scores[i] = ns["pose_metrics"](p, g)
dtws = np.array([scores[i]["dtw_mje"] for i in sample])
pick_dtw = scores[pick]["dtw_mje"]
pct = 100 * float((dtws < pick_dtw).mean())
print(f"chosen test sentence index {pick} (pick seed {args.pick_seed}): DTW-MJE {pick_dtw:.3f}; "
      f"sample of {len(sample)} held-out sentences: median {np.median(dtws):.3f}, "
      f"range {dtws.min():.3f}-{dtws.max():.3f}; chosen is better than {100 - pct:.0f}% of them")

pred, gt = generate(pick)
rec = test_ds.records[pick]
gloss = rec["gloss"]
meta = {
    "sentence_id": rec["sentence_id"], "gloss": gloss, "test_index": pick, "pick_seed": args.pick_seed,
    "generated_frames": int(len(pred)), "real_frames": int(len(gt)),
    "dtw_mje": float(pick_dtw), "motion_ratio": float(scores[pick]["motion_ratio"]),
    "pck_0.5": float(scores[pick]["PCK@0.5"]),
    "context": {"n": len(sample), "median_dtw_mje": float(np.median(dtws)),
                "min": float(dtws.min()), "max": float(dtws.max())},
    "checkpoint_epoch": int(ck["epoch"]) + 1,
}
json.dump(meta, open(os.path.join(args.out_dir, "sign_video_meta.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
np.save(os.path.join(args.out_dir, "generated_poses.npy"), pred)
np.save(os.path.join(args.out_dir, "real_poses.npy"), gt)

# ---------------------------------------------------------------- render
import cv2
import imageio

HAND = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
        (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
BODY = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
HEAD_PTS = [0, 2, 5, 7, 8, 9, 10]
LHAND, RHAND = 33, 54            # 21 landmarks each
draw_idx = list(range(0, 17)) + list(range(33, 75))

allp = np.concatenate([gt[:, draw_idx].reshape(-1, 3), pred[:, draw_idx].reshape(-1, 3)])
x0, x1 = np.percentile(allp[:, 0], [0.5, 99.5])
y0, y1 = np.percentile(allp[:, 1], [0.5, 99.5])
cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
half = max(x1 - x0, y1 - y0) / 2 + 0.35
P = 480                                   # panel size in pixels
sc = P / (2 * half)


def px(pt):
    return int((pt[0] - cx) * sc + P / 2), int((pt[1] - cy) * sc + P / 2)


def draw_panel(frame, title):
    img = np.full((P, P, 3), 30, np.uint8)
    for a, b in BODY:
        cv2.line(img, px(frame[a]), px(frame[b]), (200, 200, 200), 3, cv2.LINE_AA)
    for k in HEAD_PTS:
        cv2.circle(img, px(frame[k]), 3, (230, 230, 230), -1, cv2.LINE_AA)
    cv2.line(img, px(frame[7]), px(frame[8]), (120, 120, 120), 1, cv2.LINE_AA)
    for base, col in ((LHAND, (60, 160, 255)), (RHAND, (255, 190, 60))):          # BGR
        for a, b in HAND:
            cv2.line(img, px(frame[base + a]), px(frame[base + b]), col, 2, cv2.LINE_AA)
        for k in range(21):
            cv2.circle(img, px(frame[base + k]), 3, (255, 255, 255), -1, cv2.LINE_AA)
    for k in (11, 12, 13, 14, 15, 16):
        cv2.circle(img, px(frame[k]), 5, (200, 200, 200), -1, cv2.LINE_AA)
    cv2.putText(img, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def caption_strip(width, height=64):
    strip = np.full((height, width, 3), 20, np.uint8)
    if args.caption_png and os.path.exists(args.caption_png):
        im = cv2.imread(args.caption_png)
        if im is not None:
            h, w = min(im.shape[0], height), min(im.shape[1], width)
            strip[:h, :w] = im[:h, :w]
            return strip, True
    try:
        from PIL import Image, ImageDraw, ImageFont, features
        font_path = next(iter(glob.glob(r"C:\Windows\Fonts\Nirmala*.ttf*")), None)
        if font_path is None or not features.check("raqm"):
            return strip, False
        font = ImageFont.truetype(font_path, 30, layout_engine=ImageFont.Layout.RAQM)
        im = Image.fromarray(strip)
        ImageDraw.Draw(im).text((16, 12), gloss, font=font, fill=(255, 255, 255))
        return np.array(im), True
    except Exception:
        return strip, False


T = max(len(gt), len(pred))
strip, has_caption = caption_strip(2 * P)
out_mp4 = os.path.join(args.out_dir, "nat_mse_sign_video.mp4")
writer = imageio.get_writer(out_mp4, fps=args.fps, codec="libx264", quality=8,
                            macro_block_size=1, pixelformat="yuv420p")
sheet_frames = []
sheet_at = set(np.linspace(0, T - 1, 6).astype(int).tolist())
for t in range(T):
    left = draw_panel(gt[min(t, len(gt) - 1)], "Real recording (held-out)")
    right = draw_panel(pred[min(t, len(pred) - 1)], "NAT+MSE (generated)")
    row = np.concatenate([left, right], axis=1)
    cv2.putText(row, f"frame {t + 1}/{T}", (2 * P - 150, P - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (180, 180, 180), 1, cv2.LINE_AA)
    frame = np.concatenate([row, strip], axis=0)
    writer.append_data(frame[:, :, ::-1])
    if t in sheet_at:
        sheet_frames.append(frame)
writer.close()

sheet = np.concatenate([np.concatenate(sheet_frames[:3], axis=0), np.concatenate(sheet_frames[3:], axis=0)], axis=1)
cv2.imwrite(os.path.join(args.out_dir, "nat_mse_contact_sheet.png"), sheet)
print(f"wrote {out_mp4} ({T} frames at {args.fps} fps, caption rendered: {has_caption})")
print("wrote contact sheet and sign_video_meta.json")
