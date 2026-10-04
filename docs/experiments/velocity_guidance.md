# Velocity weight x guidance scale ablation

> "Run 3" below is the primary three-model comparison run (seed 42, 452-500 epochs per model); see [results.md](../results.md).

**Why.** The architecture comparison found the diffusion generator over-moves (motion ratio 1.60,
61% of videos above 1.5) and loses to plain-MSE regression on every pose-accuracy metric. Two settings
are the obvious suspects: the velocity term (`VELOCITY_WEIGHT = 10`, training time) and
classifier-free guidance (`CFG_SCALE = 2.5`, sampling time). This ablation tests each.

**Design.**
- Guidance acts only at sampling time, so each trained checkpoint is scored at guidance scales
  1.0, 1.5, 2.5 and 4.0 (notebook Section 16e): a few minutes, no retraining.
- The velocity weight acts at training time, so it needs separate runs: 0 and 3 (trained here) and
  10 (the Run 3 checkpoint, evaluated only).
- Everything else matches Run 3: split, seed 42, early-stopping patience 150, only the diffusion
  model trained. A new metric, `jitter_ratio` (mean frame-to-frame acceleration, generated /
  reference), separates high-frequency noise from large smooth motion.
- How to read it: if velocity 0 still over-moves, the velocity term is not the cause. If motion
  ratio falls toward 1.0 with lower guidance while DTW-MJE does not improve, the over-motion is real
  but is not what costs accuracy.

**Notebooks and data.** `notebooks/experiments/velocity_guidance_vel0.ipynb`, `..._vel3.ipynb`,
`..._vel10_eval_only.ipynb` (built by `scripts/build_notebooks.py`); figure
`figures/velocity_guidance.png`; data in `results/ablation/`.

**Limits.**
- Single seed and single signer; validation loss is noisy (std about 0.3), so only large effects count.
- Training time was capped at 2.8 h per run (about 420 epochs) for compute reasons; the Run 3
  diffusion model trained 452 epochs.

## Results (run 2026-10-02, seed 42, 192 test sentences, one signer)

Velocity weights 0 (470 epochs, time-budget stop, best-val checkpoint epoch 330), 3 (411
epochs, early stop, best 261) and 10 (the Run 3 checkpoint, evaluated only; 452 epochs,
best 302 - its `epochs_trained` shows 0 in the CSV because history is empty in eval-only
mode). Same checkpoint scored at four guidance scales. For reference:
Progressive Transformer DTW-MJE 0.195, NAT + MSE 0.112.

| Velocity | CFG | Motion ratio | Jitter ratio | DTW-MJE | PCK@0.2 | PCK@0.5 |
|---|---|---|---|---|---|---|
| 0 | 1.0 | 0.555 | 1.31 | 0.623 | 11.8% | 50.2% |
| 0 | 1.5 | 0.990 | 2.45 | 0.497 | 22.7% | 62.4% |
| 0 | 2.5 | 1.475 | 3.73 | 0.391 | 35.3% | 73.8% |
| 0 | 4.0 | 1.614 | 4.09 | 0.355 | 40.0% | 78.9% |
| 3 | 1.0 | 0.816 | 2.04 | 0.542 | 17.3% | 58.7% |
| 3 | 1.5 | 1.268 | 3.22 | 0.416 | 30.4% | 71.0% |
| 3 | 2.5 | 1.604 | 4.14 | 0.345 | 39.8% | 79.0% |
| 3 | 4.0 | 1.646 | 4.27 | 0.335 | 42.3% | 80.5% |
| 10 | 1.0 | 0.596 | 1.43 | 0.642 | 11.2% | 48.1% |
| 10 | 1.5 | 1.110 | 2.78 | 0.487 | 23.1% | 63.4% |
| 10 | 2.5 | 1.606 | 4.07 | 0.357 | 38.2% | 77.3% |
| 10 | 4.0 | 1.690 | 4.29 | 0.331 | 42.1% | 81.2% |

The velocity-10 / CFG-2.5 row reproduces Run 3 (DTW-MJE 0.357, motion 1.60), so the
evaluation path is consistent. Figure: `figures/velocity_guidance.png`;
tidy table: `results/ablation/velocity_guidance_grid.csv`.

**What it shows**

1. **Guidance is a large, monotonic effect; velocity weight is not.** Raising CFG from 1.0
   to 4.0 moves DTW-MJE from 0.54-0.64 to 0.33-0.36 and motion ratio from 0.56-0.82 to
   1.61-1.69, in all three runs. Across velocity weights at a fixed guidance the differences
   are small and non-monotonic (DTW-MJE at CFG 2.5: 0.391 / 0.345 / 0.357 for 0 / 3 / 10;
   motion ratio at CFG 1.0: 0.555 / 0.816 / 0.596). That ordering is what single-seed run-to-
   run noise looks like, so there is no evidence here that the velocity weight matters for
   these metrics.
2. **Turning guidance down does not fix the over-motion usefully.** Motion ratio only
   reaches ~1.0 around CFG 1.0-1.5, where DTW-MJE is 0.42-0.64, i.e. worse than at the
   default. Guidance trades accuracy against motion in the wrong direction for this goal.
3. **The extra motion is mostly high-frequency noise.** Jitter ratio (mean acceleration,
   pred/gt) is 1.3-4.3 and exceeds the motion ratio in every cell; at CFG 2.5 it is about
   2.5x the motion ratio. Ground-truth-like motion would have both near 1.0.
4. **No setting gets near the other two models.** The best cell is DTW-MJE 0.331 (velocity 10,
   CFG 4.0), 70% above Progressive's 0.195 and about 3x NAT+MSE's 0.112.

**Caution: the velocity term is not shown to be what prevents collapse.** With the velocity term
removed entirely (weight 0) the model does not collapse to a frozen output: motion ratio is 1.48 at
guidance 2.5 (0.56 at guidance 1.0). An early run without the velocity term did collapse (motion ratio
about 0.02) and motivated adding it, but that early comparison also differed in other settings
(`BONE_WEIGHT` 1.5 vs 20, the split, leg weighting, guidance), so it was never a clean ablation. Either
the early collapse had another cause, or something in the current configuration prevents it; this
ablation cannot say which. It shows only that removing velocity alone does not reproduce the collapse
today. The claim that the velocity term resolves a collapse is therefore not established; settling it
would need weight 0 re-run with the earlier settings (for example `BONE_WEIGHT = 1.5`).

**Not tested (hypotheses only).** The jitter pointing at the diffusion sampler (50 DDIM
steps, x0 prediction, guidance extrapolation) rather than the loss is an inference from
this grid, not something these runs isolate. Cheap inference-only follow-ups on the saved
checkpoint: sweep `SAMPLING_STEPS`; measure DTW-MJE after stronger temporal smoothing than
the current 7-frame `PoseRefiner` average. A velocity-1 run was planned but not executed; given point 1 it is not a priority.

## Follow-up: test-time smoothness guidance (run 2026-10-02)

Motivated by the jitter finding above and by MotionFlow (Meral, Yesiltepe, Dunlop, Yanardag,
"MotionFlow: Attention-Driven Motion Transfer in Video Diffusion Models", AAAI 2026), which
steers a pretrained video diffusion model without training by taking gradient steps on the
noisy latents during the first part of sampling (20 of 50 steps there), using a loss that
includes a temporal-consistency term. MotionFlow works on video latents with attention-
derived masks; only the *mechanism* (training-free latent guidance on a temporal-smoothness
loss) is borrowed here. It is not a reimplementation of their method.

- `DiffusionG2P._smooth_step`: at the first `smooth_steps` sampling steps, `smooth_iters`
  normalised gradient steps on x_t reduce the squared frame-to-frame acceleration of the
  implied x0 (padding masked out). No weights change; checkpoint is the Run 3 best.
- Section 16f arms, all at CFG 2.5: default 7-frame refiner; **controls** with a 15- and
  31-frame refiner (plain smoothing, no gradients); guidance at lr 0.02 / 0.05 / 0.10; and
  guidance + 15-frame refiner. If guidance does not beat plain smoothing on validation it
  is not worth its extra sampling cost.
- Protocol: choose the arm on the **validation** set, report the test set once (the test set
  has already been looked at many times in this project). All arms' test numbers are saved
  in `smooth_guidance_arms.csv`; report all of them, not just the winner.
- Notebook: `notebooks/experiments/smoothness_guidance_eval_only.ipynb` (no training; loads the Run 3 diffusion checkpoint).
- Expectation, stated in advance: this can reduce jitter and may lower DTW-MJE somewhat; it
  is unlikely to close the gap to NAT+MSE (DTW-MJE 0.112 vs 0.331 at best), because
  PCK@0.5 of 77% means large systematic errors, not just high-frequency noise.

### Results (Run 3 best checkpoint, CFG 2.5, 192 test / 197 val videos, single seed)

| Arm | Val DTW-MJE | Test DTW-MJE | Test motion | Test jitter | Test PCK@0.5 | Sampling (192 videos) |
|---|---|---|---|---|---|---|
| **default (7-frame refiner)** | **0.3605** | **0.3576** | 1.602 | 4.049 | 77.44% | 27 s |
| control: refiner 15 | 0.3891 | 0.3906 | 0.835 | 2.032 | 73.74% | 27 s |
| control: refiner 31 | 0.4331 | 0.4353 | 0.499 | 1.125 | 69.31% | 27 s |
| guidance lr 0.02 | 0.4719 | 0.4643 | 0.617 | 1.348 | 66.34% | 63 s |
| guidance lr 0.05 | 0.4369 | 0.4368 | 0.608 | 1.183 | 69.56% | 63 s |
| guidance lr 0.10 | 0.4002 | 0.4049 | 0.883 | 1.511 | 73.47% | 64 s |
| guidance lr 0.05 + refiner 15 | 0.4641 | 0.4749 | 0.335 | 0.557 | 65.59% | 63 s |

The default arm reproduces Run 3 (DTW-MJE 0.358, motion 1.60), so the harness is consistent.
Selection by lowest validation DTW-MJE picks the **default** arm (no guidance, no extra
smoothing); its test numbers are the first row. All arms are listed above, not only the winner.

**Verdict: it did not help, and the expectation stated in advance only half held.**

1. **Guidance does reduce jitter** (4.05 -> 1.2-1.5 with the stronger settings), but **every
   guidance arm has worse DTW-MJE and PCK than doing nothing** (0.40-0.47 vs 0.358). Lowering
   the jitter ratio is not translating into accuracy.
2. **Guidance adds nothing over plain smoothing.** At matched jitter, a wider refiner is as
   good: refiner 31 (jitter 1.13, DTW-MJE 0.435) vs guidance lr 0.05 (jitter 1.18, 0.437);
   refiner 15 (jitter 2.03, 0.391) is slightly better than guidance lr 0.10 (jitter 1.51,
   0.405) in accuracy per unit of jitter removed. Guidance costs ~2.3x the sampling time (63 s
   vs 27 s) for no gain, so it should not be used or claimed as a contribution.
3. **The jitter is not what is costing accuracy.** If high-frequency noise were the main
   error, smoothing would lower DTW-MJE; it raises it monotonically. Part of the apparent
   over-motion is real high-frequency path length (motion ratio drops 1.60 -> 0.50 from
   smoothing alone), but smoothing also blurs genuine fast hand movement, which the metric
   punishes. The remaining error is systematic, consistent with PCK@0.5 staying ~70-77%.
4. **Nothing approaches the other two models** (best 0.358 vs 0.195 and 0.112).
5. Expected in advance: "can reduce jitter ... unlikely to close the gap". Reduced jitter: held.
   "May lower DTW-MJE somewhat": did not hold; it raised it. Single seed; the differences
   are large (0.04-0.12) relative to the val/test agreement (~0.003), so this is not noise.

Not tried, so no claim: guidance on a different loss (bone length, or MSE-like pull toward a
regressor's output), more sampling steps, or a smaller guidance scale in training. The
borrowed mechanism (MotionFlow-style test-time latent guidance) was a reasonable idea that
this experiment does not support for this model.
