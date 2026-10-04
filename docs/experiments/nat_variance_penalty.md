# NAT + temporal-variance penalty (refinement of the final system)

**Status: complete. Trained at weights 0 (control), 2 and 10. The hypothesis is not supported: the
penalty raised the motion ratio by only about 0.011 (0.818 to 0.829), far short of the 0.9-1.1 target,
with no accuracy gain.**

> "Run 3" below is the primary three-model comparison run (seed 42, 452-500 epochs per model); see [results.md](../results.md).

## Why this model

The architecture comparison (see [results.md](../results.md)) found the plain-MSE
non-autoregressive model (NAT + MSE) is by far the most accurate (DTW-MJE 0.112, PCK@0.5
99.1%) but under-moves: motion ratio 0.82. Regression toward the mean pose flattens
trajectories. The diffusion model has the opposite problem (1.60) and loses on accuracy.
The natural candidate is NAT + MSE plus a term that restores the lost temporal variance.

## What is implemented

`MSEPoseLoss(variance_weight=W)` in `scripts/build_notebook.py`:

    loss = masked MSE  +  duration loss  +  W * mean_j,c ( std_t(pred) - std_t(target) )^2

`std_t` is the standard deviation over valid frames of each joint coordinate, per sequence;
extrapolated leg landmarks are down-weighted by `LEG_WEIGHT` as elsewhere. `W = 0` is exactly
plain NAT + MSE, so the existing run (DTW-MJE 0.1122, seed 42) is the control, from the same code.
Everything else (encoder, duration predictor, length regulator, split, seed, patience 150,
500-epoch cap) is unchanged.

Notebooks: `notebooks/experiments/nat_variance_penalty_w2.ipynb` and `..._w10.ipynb`; weight 0 is
`notebooks/02_final_system_nat_mse.ipynb`. (Built by `scripts/build_notebooks.py`.)

Section 16g writes `nat_variance_w<W>.csv` (val, test, cross-signer rows; motion ratio,
jitter ratio, DTW-MJE, PCK) and a per-sample test CSV.

## Pre-registered success criteria (written before any run)

The hypothesis is that the penalty fixes under-motion **without** costing accuracy. It
counts as supported only if, for the weight chosen on **validation**:

1. test motion ratio is within 0.9-1.1 (plain MSE: 0.82), and
2. test DTW-MJE is no more than 10% worse than plain NAT + MSE (<= 0.123), and
3. jitter ratio does not blow up relative to plain NAT + MSE (a penalty can raise motion ratio by
   adding noise rather than real motion; if jitter rises much more than motion, that is
   gaming the metric, not a fix).

If it fails any of these, report it as a negative result. Beating plain NAT + MSE on DTW-MJE or
PCK is **not** the criterion and is not expected: plain NAT + MSE is already near the ceiling on
those (PCK@0.5 99.1%), so the honest best case is "same accuracy, correct amount of motion".

## Interpretation notes

- This model was designed **after** seeing test-set results for the three-model comparison. State that it
  is a post-hoc, exploratory improvement motivated by an observed weakness, not a pre-planned
  hypothesis. Choose W on validation only; quote the test set once; list every weight tried.
- Single seed, single signer. Report cross-signer numbers as well.
- The motion ratio is a path-length metric and can be gamed; always report jitter ratio and
  look at the videos.
- DTW-MJE/PCK still do not measure whether the signing is understandable. Back-translation
  and human evaluation remain the open measurements.
- If it works, the contribution is modest and real: "a plain regression model with a variance
  term matches the accuracy of plain regression while restoring natural amount of motion,
  and beats a diffusion model on both." If it does not, the contribution is the diagnosis.

## Interim results (seed 42, 192 test / 197 val videos, one signer)

Both weights trained to the end (epochs 492 and 500; best-validation checkpoints at 342 and
458). Reference: earlier plain NAT + MSE, i.e. plain MSE, test motion 0.820, DTW-MJE 0.1122,
PCK@0.5 99.13% (its jitter ratio was not recorded; the W=0 control supplies it).

| Weight | Split | Motion ratio | Jitter ratio | DTW-MJE | PCK@0.5 |
|---|---|---|---|---|---|
| 2 | val | 0.819 | 0.764 | 0.1087 | 99.32% |
| 2 | **test** | 0.829 | 0.775 | 0.1122 | 99.12% |
| 2 | cross-signer | 0.660 | 0.690 | 0.3052 | 85.42% |
| 10 | val | 0.822 | 0.768 | 0.1088 | 99.31% |
| 10 | **test** | 0.829 | 0.781 | 0.1132 | 99.09% |
| 10 | cross-signer | 0.660 | 0.696 | 0.3051 | 85.43% |

Against the pre-registered criteria:

1. **Motion ratio within 0.9-1.1 - failed** at both weights, on validation (0.82) and test
   (0.83). The penalty did not move it from the unpenalised 0.82.
2. DTW-MJE within 10% of plain NAT + MSE - met (0.112 / 0.113 vs 0.1122), but trivially, because
   nothing changed.
3. Jitter not blown up - met: jitter ratio is 0.78 (output smoother than the reference).

The two weights are indistinguishable (val DTW-MJE 0.1087 vs 0.1088, motion 0.819 vs 0.822),
so which weight validation selects does not matter: both fail criterion 1. The hypothesis,
that this penalty fixes under-motion without costing accuracy, is **not supported**, and
there is no accuracy gain either. This model should not be presented as a contribution.

## Control (W = 0, same code) and final verdict

The W=0 run (plain MSE, 500 epochs, best-validation checkpoint 386) reproduces the earlier
plain NAT + MSE almost exactly (test DTW-MJE 0.1113 vs 0.1122; motion 0.818 vs 0.820), so two
independent plain-MSE runs differ by about 0.002 in motion ratio. It also supplies the jitter
ratio the earlier run lacked.

| Weight | Split | Motion ratio | Jitter ratio | DTW-MJE | PCK@0.5 |
|---|---|---|---|---|---|
| 0 (control) | val | 0.811 | 0.759 | 0.1076 | 99.35% |
| 0 (control) | **test** | 0.818 | 0.770 | 0.1113 | 99.10% |
| 0 (control) | cross-signer | 0.651 | 0.687 | 0.3045 | 85.45% |
| 2 | **test** | 0.829 | 0.775 | 0.1122 | 99.12% |
| 10 | **test** | 0.829 | 0.781 | 0.1132 | 99.09% |

Paired per-video differences on the 192 test videos, penalty minus control (Wilcoxon):

| Comparison | Motion ratio | DTW-MJE | Jitter ratio |
|---|---|---|---|
| weight 2 - w0 | +0.0108 (p = 2e-6) | +0.0009 (p = 0.21, n.s.) | +0.0045 (p = 0.44, n.s.) |
| weight 10 - w0 | +0.0111 (p = 1e-5) | +0.0020 (p = 0.003) | +0.0106 (p = 0.07, n.s.) |

**The penalty is not exactly inert, but its effect is negligible.** It does move motion ratio in
the intended direction (+0.011, 61% of videos, both weights, about five times the ~0.002
difference seen between two plain-MSE runs), so this is probably a real effect. But it is
roughly 1% of what would be needed (0.82 -> 0.9+), and weight 10 costs a statistically
detectable but practically tiny amount of accuracy (DTW-MJE +0.002, ~1.8% worse). These
p-values are paired within one training run each; they do not capture seed-to-seed training
variance (one seed per weight).

Pre-registered criteria, final:

1. Motion ratio within 0.9-1.1: **failed** (test 0.829 at both weights; val 0.819 / 0.822).
2. DTW-MJE within 10% of the control (<= 0.1224): met (0.1122 / 0.1132), at a tiny cost.
3. Jitter not blown up relative to the control: met (0.775 / 0.781 vs 0.770).

**Verdict: the hypothesis (this penalty fixes under-motion without costing accuracy) is not
supported.** There is no accuracy gain, the motion shift is negligible, and the model should not
be presented as a contribution. The explanation in the next section (the reference's own
frame-level jitter, which no gloss-conditioned model can predict, caps motion ratio near 0.8)
accounts for why a penalty on low-frequency variance cannot move it.

## Why it did nothing: the reference motion is largely high-frequency

`scripts/gt_motion_smoothing.py` smooths the *ground-truth* recordings with a
moving average and measures how much path length survives (40 recordings, signer U4):

| Window | GT path length kept | GT acceleration kept |
|---|---|---|
| 3 frames | 0.909 | 0.735 |
| 5 frames | 0.809 | 0.521 |
| 7 frames | 0.726 | 0.377 |
| 15 frames | 0.528 | 0.186 |

A 5-frame average alone removes 19% of the reference's own path length, leaving 0.81, almost
exactly plain NAT + MSE's motion ratio (0.82). So the "under-motion" of a deterministic
regressor appears to be mostly the loss of frame-level jitter in the reference (tracker noise
or fine finger movement), which nothing conditioned only on the gloss can predict. A penalty
on temporal *standard deviation* acts on low-frequency amplitude, which the model already
matches, so it has nothing to fix. Consequences, stated with the limits of this evidence
(one signer, 40 recordings, a correlation with 0.82, not a controlled test):

- A motion ratio around 0.8 for a smooth, accurate regressor is probably not a defect, and
  chasing 1.0 would mean generating noise. The diffusion model's 1.60 (jitter 4x the
  reference) is the opposite error: it adds noise on top of the real motion.
- `motion_ratio` as defined (raw path length) is contaminated by reference jitter. The
  frozen-output flag (< 0.5) is still meaningful; reading 0.8 as "under-moving" is not. A
  fairer motion metric would compute both sides on smoothed trajectories.
