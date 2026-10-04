# Methodology

## Data

- 1,922 Bangla sentences × 5 signers = 9,610 recordings. Each is a `(frames, 95, 3)` array: 33 body,
  42 hand (21 per hand) and 20 further landmarks, 117–290 frames (median 170).
- Coordinates are already normalised so that median shoulder width is 1.0; all pose errors are therefore
  in **shoulder-widths**. A data-quality check (NaN/Inf, all-zero sequences, near-zero shoulder width,
  frame-to-frame jumps) runs before training and found no blocking issue.
- **Landmarks 23–32 (hips, legs, feet) are extrapolated, not observed**: left/right medians agree to
  0.01 and shoulder-to-hip distance is 2.40 shoulder-widths against about 1.25 anatomically. They carry
  the largest coordinate magnitudes, and because Soft-DTW works on the flattened vector they dominated
  the objective (hands received about 11% of it). Weighting them 0.1 raises the hands' share to about 84%.
  The recogniser and the video renderer ignore them.
- **Vocabulary.** The gloss language is a fixed template (kinship term, time, food, verb, negation)
  over only **28–29 distinct tokens**, sentences of 3–7 tokens, 1,785 distinct gloss strings among 1,922
  sentence IDs. Consequences: chance level for token-based metrics is high, so every back-translation
  result is reported against a wrong-sentence floor and a real-pose ceiling; and glosses repeat across
  sentence IDs (below).
- Signer identity is parsed from the filename token `_U<n>_` (U4, U5, U6, U7, U19) and mapped to
  indices by numeric order. Sorted filename position is wrong because string sort puts U19 before U4.

## Splits

- 80/10/10 over **normalised gloss text** (`DEDUP_BY_GLOSS`): whitespace collapsed, sentence-final danda
  removed. Five signers record each sentence and the same gloss recurs under different sentence IDs, so
  splitting by recording or by ID leaks identical sentences into the test set (on an ID-disjoint split,
  15.1% of test sentences had a training gloss at cosine similarity ≥ 0.95, four verbatim identical).
- The partition is decided before signer filtering, so single-signer and all-signer runs hold out the
  same sentences. Disjointness is asserted in the notebook.
- Main experiment: train on one signer (U4, index 0); validate and test on that signer's held-out
  sentences (197 val / 192 test videos). The other four signers record the same held-out sentences and
  form the cross-signer test (768 videos): the model generates as the training signer and is scored
  against each unseen signer's recording.

## Models

All three share the BanglaBERT text encoder (`csebuetnlp/banglabert`, fine-tuned, projected to d=256,
plus two Transformer encoder layers) and are trained from scratch in the same loop.

- **NAT + MSE (final system).** Duration predictor (two 1-D conv layers, regresses log(1+frames) per
  token) → length regulator (repeats each token's encoding for its duration; training uses
  ground-truth durations allocated uniformly across tokens, inference uses predicted ones) →
  sinusoidal positions → four-layer Transformer encoder → linear head to 95×3 coordinates. Loss: masked
  per-frame MSE + duration MSE (log domain). No Soft-DTW, velocity or bone term.
- **Diffusion (NAT + Diffusion + Velocity).** Same encoder, duration predictor and length regulator;
  the head is a four-layer Transformer decoder denoiser predicting x0, 250 training timesteps, 50
  sampling steps, classifier-free guidance (condition dropout 0.1, scale 2.5), followed by a 7-frame
  moving-average pose refiner. Loss: length-normalised Soft-DTW (γ=0.1) + bone-length term (weight 20)
  + velocity term (weight 10) + duration term, with hands up-weighted (1.5) and extrapolated legs
  down-weighted (0.1).
- **Progressive Transformer** (Saunders et al., ECCV 2020 style). Autoregressive causal decoder that
  predicts the next pose frame and a progress counter in [0, 1]; generation stops when the counter
  saturates, so there is no duration predictor. Loss: masked MSE on pose and counter. Training is one
  parallel pass (teacher forcing); inference is sequential.

Parameter counts including BanglaBERT: diffusion 118.8 M, Progressive 116.2 M, NAT + MSE 115.4 M.

## Training protocol (identical for all models)

AdamW, learning rate 2e-4 for new layers and 3e-5 for BanglaBERT, weight decay 1e-4, batch size 16 with
length-grouped batches (padding below 1%), 10-epoch linear warmup then cosine decay, gradient clipping 1.0,
mixed precision with the loss in fp32, up to 500 epochs, early stopping on an exponential moving average
(0.8) of validation loss with patience 150, best-validation checkpoint reported, seed 42. The diffusion
model's early run used patience 60 and stopped at epoch 210; the reported runs use patience 150
(see [results.md](results.md)).

## Metrics

All pose metrics are computed **along a DTW alignment path** between generated and reference sequences,
so duration error is not charged to the pose model (on a motion and a 1.5× slowed copy of itself, naive
frame-index error is 0.637 while DTW-MJE is 0).

- **DTW-MJE** – mean joint error along the path (shoulder-widths).
- **PCK@t** – percentage of aligned joint pairs within t shoulder-widths (t = 0.05, 0.1, 0.2, 0.5).
- **Length ratio** – generated / reference frame count (isolates the duration predictor).
- **Motion ratio** – generated / reference travelled distance (a value below 0.5 flags a frozen output).
  It is a raw path-length ratio and so is contaminated by frame-level jitter in the reference
  ([results.md](results.md)).
- **Jitter ratio** – generated / reference mean frame-to-frame acceleration (mean second difference);
  separates high-frequency noise from large smooth motion.
- **Back-translation** – see below.

## Statistics

Comparisons use the same 192 test sentences for every model, so tests are paired per video. Because the
test set holds one signer and one recording per distinct gloss, videos are independent units. Reported:
paired Wilcoxon signed-rank tests on per-video DTW-MJE, and 95% bootstrap confidence intervals
(resampling sentences, 2,000 resamples) for back-translation metrics. Experiments with decision rules
(variance penalty, back-translation) were pre-registered in the experiment write-ups before any result
existed. Single seed throughout; seed-to-seed training variance is not measured.

## Back-translation

A CTC Transformer recogniser (pose → gloss tokens) is trained **only on real training poses** (all five
signers, same gloss-level split) with smoothing, speed and noise augmentation, and early-stopped on real
validation poses. It reads each model's generated test poses; sentence exact-match accuracy, word error
rate and BLEU-1/2 are reported with floor (wrong-sentence) and ceiling (real-pose) conditions. A validity
gate fixed in advance requires the recogniser to read at least 90% of real test sentences exactly; it read
81.8%, so the study is reported as exploratory. See [experiments/back_translation.md](experiments/back_translation.md).
