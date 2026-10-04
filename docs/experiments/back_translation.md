# Back-translation evaluation

> "Run 3" below is the primary three-model comparison run (seed 42, 452-500 epochs per model); see [results.md](../results.md).

**Status (2026-10-03): run on Kaggle. The pre-registered validity gate was
NOT met, so by the rules below the model comparison is exploratory, not a clean result.**
The numbers are reported in full under "Results", with what the gate failure does and does not
undermine.

## Why

DTW-MJE and PCK measure geometric closeness to one recorded performance. They cannot say
whether generated signing carries the *meaning*, and the project's own findings show they can
mislead (a smooth regressor scores near-perfectly; smoothing the output makes the score
worse even though smoothing removes noise). Back-translation asks the question that matters:
can the generated signing be read back as the sentence it came from?


## What is implemented (`scripts/build_notebook.py`, Sections 16h-16j)

- **Recogniser.** CTC transformer (4 layers, d=256) over pose sequences -> gloss tokens.
  Trained on **real training poses only**, all five signers, using the same gloss-text split
  (no test or validation sentence, nor any identical gloss, is seen in training). Early-stopped
  on real validation poses. Legs (extrapolated landmarks) are excluded. Augmentation:
  random smoothing (windows 1-15), speed change (0.85-1.15x), Gaussian noise (up to 0.02
  shoulder-widths), so it is not unfairly harsh on smooth or jittery output.
- **Generation.** The Run 3 checkpoints (`g2p_best.pt`, `g2p_progressive_best.pt`,
  `g2p_nat_mse_best.pt` from the main run) each generate the 192 test
  sentences (single signer, one recording per distinct gloss). The diffusion model is scored at
  CFG 1.0 / 2.5 / 4.0 (guidance is conditioning strength, so it may matter for meaning even
  though it did not help geometry).
- **Conditions scored.** Real test poses (ceiling); real poses smoothed 7- and 15-frame
  (does smoothing alone hurt recognition?); Progressive Transformer; NAT+MSE; diffusion x3;
  and a wrong-sentence floor (the real-pose hypotheses scored against other sentences' glosses,
  averaged over 50 random pairings).
- **Metrics.** Sentence exact-match accuracy, token WER, BLEU-1/2 (sentences are 3-7 tokens,
  so BLEU-3/4 are not meaningful), 95% bootstrap CIs resampling sentences, and paired bootstrap
  differences in WER between models.
- Notebook: `notebooks/03_back_translation.ipynb` (needs the three trained checkpoints). Outputs:
  `results/back_translation/summary.csv` and `per_sentence_scores.csv` (gloss text omitted from the
  public copy).

## Pre-registered rules (written before any run)

1. **Validity gate.** The recogniser must read >= 90% of *real* test sentences exactly. If it
   does not, the comparison is reported as uninformative, not as a model result. The smoothed-real
   rows must also not collapse (a drop there means the recogniser punishes smoothing, which
   would bias against smooth models, and must be reported).
2. **Reference points.** A model is only "readable" relative to the ceiling (real poses) and
   the floor (wrong-sentence). With 29 gloss tokens in a fixed template, the floor is far above
   zero for token accuracy; quote exact-match and WER against it.
3. **No predicted winner.** The order of Progressive / NAT+MSE / diffusion is whatever is
   measured, with CIs. A difference counts only if its paired bootstrap CI excludes zero.
4. The checkpoints were chosen on geometric validation loss, not recognition; this is a
   limitation, not something to tune against. Do not re-pick checkpoints or guidance to
   improve this score and then quote it as an unbiased result.

## Limits

- The corpus is a 29-token template language; recognition is a much easier problem than
  intelligibility to a Deaf signer, so this is a necessary-not-sufficient check. Human evaluation
  with BdSL signers (Gap 5) is still the real measure.
- A recogniser trained on real poses can fail on generated poses because of distribution
  shift, not because the meaning is missing. The smoothed-real rows and the augmentation are
  partial mitigations only.
- Single signer, single seed, 192 sentences; the recogniser is one trained model.

## Results (seed 42, 192 test sentences, one signer, Run 3 checkpoints)

Recogniser: 28 training tokens, trained on 7,665 real recordings (5 signers), early-stopped
at epoch 62; on 985 real validation recordings it reads 89.0% of sentences exactly (WER 2.2%).

| Condition | Sentence exact % [95% CI] | WER % [95% CI] | BLEU-1 | BLEU-2 |
|---|---|---|---|---|
| **Real test poses (ceiling)** | 81.8 [76.0, 87.0] | 3.6 [2.6, 4.7] | 96.5 | 95.9 |
| Real poses, smoothed 7-frame | 83.3 [78.1, 88.0] | 3.4 [2.3, 4.5] | 96.7 | 96.1 |
| Real poses, smoothed 15-frame | 75.0 [68.7, 81.2] | 5.2 [3.9, 6.6] | 95.0 | 93.4 |
| NAT + MSE | 67.2 [60.9, 73.4] | 7.5 [5.9, 9.2] | 93.0 | 91.7 |
| Progressive Transformer | 11.5 [7.3, 16.1] | 42.8 [39.1, 46.3] | 60.3 | 48.3 |
| Diffusion, CFG 4.0 | 0.0 [0.0, 0.0] | 82.2 [80.0, 84.5] | 19.9 | 10.0 |
| Diffusion, CFG 2.5 (default) | 0.0 [0.0, 0.0] | 87.4 [85.1, 89.8] | 16.0 | 8.2 |
| Diffusion, CFG 1.0 | 0.0 [0.0, 0.0] | 96.7 [94.6, 99.1] | 3.7 | 0.7 |
| Wrong-sentence control (floor) | 0.1 [0.1, 0.2] | 83.3 [81.1, 85.8] | - | - |

Paired bootstrap differences in WER (percentage points; negative = first reads better), all
with CIs excluding zero: NAT+MSE vs Progressive -35.4 [-39.3, -31.4]; diffusion (CFG 2.5) vs
NAT+MSE +80.0 [+77.3, +82.7]; diffusion (CFG 2.5) vs Progressive +44.6 [+40.0, +49.2];
diffusion CFG 4 vs CFG 1 -14.5 [-17.5, -11.7].

## Verdict against the pre-registered rules

1. **Validity gate: NOT MET.** The recogniser reads 81.8% of real test sentences exactly
   (criterion >= 90%). By rule 1, the comparison is reported as exploratory and must not be
   presented as an established model ranking. The smoothed-real rows did not collapse (83.3%
   and 75.0%), so the recogniser does not simply punish smooth output.
2. What the gate failure does and does not undermine. At the *token* level the recogniser is
   good (WER 3.6% on real poses); sentence exact-match is lower only because one wrong token
   fails a ~5-token sentence. The gaps between models (WER 7.5% vs 42.8% vs 82-97%) are an order
   of magnitude larger than that recogniser error. That makes it unlikely the ordering is an
   artefact of recogniser imperfection. It does not address the larger threat below
   (distribution shift), and it is a judgement made after seeing the results, not licence to
   relax the gate.
3. **Exploratory reading, with the limits stated.**
   - NAT+MSE is close to real signing in readability (WER 7.5% vs 3.6% ceiling); the
     Progressive Transformer is partly readable (42.8%); the diffusion model is at or beyond
     the wrong-sentence floor (82-97% vs 83.3%), i.e. its output is not recognisable as the
     input sentence by this recogniser at any tested guidance.
   - Guidance helps (CFG 1 -> 4: WER 96.7 -> 82.2, CI excludes zero) but never clears the floor.
     The "worse than the floor" values (87-97%) mostly reflect empty/garbage hypotheses scored as
     deletions, not meaning pointing the wrong way.
   - This agrees with the geometric results (the regressors are more accurate), and adds that
     the diffusion model's large, jittery motion is not carrying readable content.
4. **The main unresolved confound: noise versus meaning.** The recogniser was trained on clean
   real poses (plus mild augmentation). The diffusion output has about 4x the reference's
   acceleration ([velocity_guidance.md](velocity_guidance.md)). It is not established whether its outputs are unreadable because
   the signing is wrong or because high-frequency noise defeats a recogniser that never saw
   noise like that. The smoothed-real rows test smoothing, not added noise. A follow-up that
   would separate the two (not yet run, and it must be pre-registered before running): (a)
   score the diffusion output after smoothing it (7- and 15-frame) - if readability recovers,
   noise was the problem; (b) add noise matched to the diffusion jitter to real poses - if
   that also destroys recognition, the recogniser is fragile rather than the model wrong.
5. Still necessary-not-sufficient evidence: a 28-token template language, a single trained
   recogniser, one signer and one seed. Human evaluation with BdSL signers remains the real
   measure.
