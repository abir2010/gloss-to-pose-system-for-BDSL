"""Build every notebook in this repository from the single generator, scripts/build_notebook.py.

    python scripts/build_notebooks.py

The notebooks are generated artefacts: edit the generator, not the .ipynb files. Each experiment
variant below is the same generator with different flags; the README maps each to its results.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "scripts" / "build_notebook.py"

# (output path, generator flags, what it is)
NOTEBOOKS = [
    ("notebooks/01_full_pipeline.ipynb", [],
     "All three models (diffusion, NAT+MSE, Progressive Transformer) trained and compared"),
    ("notebooks/02_final_system_nat_mse.ipynb", ["--nat-only", "--variance-weight", "0"],
     "The final system alone: NAT + MSE trained and evaluated (val / test / cross-signer)"),
    ("notebooks/03_back_translation.ipynb", ["--backtrans"],
     "Pose-to-gloss recogniser + back-translation of every model's output (needs trained checkpoints)"),
    ("notebooks/experiments/velocity_guidance_vel0.ipynb",
     ["--ablation", "--velocity-weight", "0", "--time-budget", "2.8"],
     "Diffusion model, velocity weight 0, then guidance sweep"),
    ("notebooks/experiments/velocity_guidance_vel3.ipynb",
     ["--ablation", "--velocity-weight", "3", "--time-budget", "2.8"],
     "Diffusion model, velocity weight 3, then guidance sweep"),
    ("notebooks/experiments/velocity_guidance_vel10_eval_only.ipynb", ["--ablation", "--eval-only"],
     "Guidance sweep on the existing velocity-10 diffusion checkpoint (no training)"),
    ("notebooks/experiments/smoothness_guidance_eval_only.ipynb", ["--ablation", "--eval-only", "--smooth"],
     "Test-time smoothness guidance vs plain smoothing on the diffusion checkpoint (no training)"),
    ("notebooks/experiments/nat_variance_penalty_w2.ipynb", ["--nat-only", "--variance-weight", "2"],
     "NAT + MSE with a temporal-variance penalty, weight 2"),
    ("notebooks/experiments/nat_variance_penalty_w10.ipynb", ["--nat-only", "--variance-weight", "10"],
     "NAT + MSE with a temporal-variance penalty, weight 10"),
]

if __name__ == "__main__":
    for rel, flags, what in NOTEBOOKS:
        out = ROOT / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(GEN), *flags, "--out", str(out)]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        status = "ok " if res.returncode == 0 else "FAIL"
        print(f"[{status}] {rel}  -  {what}")
        if res.returncode != 0:
            print(res.stdout[-800:], res.stderr[-800:])
            sys.exit(1)
