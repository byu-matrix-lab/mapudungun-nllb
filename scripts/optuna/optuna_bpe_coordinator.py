#!/usr/bin/env python3
"""
Optuna BPE coordinator — runs on the login node (in a tmux/screen session).

Searches for the optimal arn-only BPE vocabulary size that maximizes
dev-set chrF++ after fine-tuning NLLB-200 600M on the blocks/arn→es task.

Each trial:
  1. Suggests a vocab size via Bayesian search (log-uniform 1000–20000)
  2. Submits optuna_bpe_trial.slurm via sbatch --wait (blocks until done)
  3. Reads the result JSON written by the trial script
  4. Reports the chrF++ score back to Optuna

Run from the project root:
  uv run python scripts/optuna/optuna_bpe_coordinator.py
  uv run python scripts/optuna/optuna_bpe_coordinator.py --n-trials 10

Resume automatically if interrupted (SQLite study is persisted).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import optuna

PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULT_DIR  = Path("/home/it238/nobackup/autodelete/mapudungun/optuna_bpe_results")
STUDY_DB    = Path("/home/it238/nobackup/autodelete/mapudungun/optuna_bpe.db")
STUDY_NAME  = "optuna_bpe_vocab"
SLURM_SCRIPT = str(PROJECT_DIR / "configs/slurm/optuna_bpe_trial.slurm")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=12,
                   help="Number of Optuna trials (default: 15)")
    p.add_argument("--vocab-min", type=int, default=1000)
    p.add_argument("--vocab-max", type=int, default=20000)
    return p.parse_args()


def objective(trial: optuna.Trial) -> float:
    vocab_size = trial.suggest_int("vocab_size", 1000, 20000, log=True)
    trial_id   = trial.number

    result_file = RESULT_DIR / f"trial_{trial_id}.json"

    # If a previous interrupted run already produced this result, reuse it
    if result_file.exists():
        data = json.loads(result_file.read_text())
        print(f"[trial {trial_id}] Reusing cached result: "
              f"vocab_size={data['vocab_size']}  chrF++={data['chrf']:.4f}")
        return data["chrf"]

    print(f"\n[trial {trial_id}] Submitting SLURM job: vocab_size={vocab_size}")
    cmd = [
        "sbatch", "--wait",
        f"--export=ALL,TRIAL_ID={trial_id},VOCAB_SIZE={vocab_size}",
        SLURM_SCRIPT,
    ]
    proc = subprocess.run(cmd, cwd=str(PROJECT_DIR),
                          capture_output=True, text=True)
    print(proc.stdout.strip())
    if proc.returncode != 0:
        print(f"sbatch failed:\n{proc.stderr}", file=sys.stderr)
        raise optuna.exceptions.TrialPruned()

    if not result_file.exists():
        print(f"ERROR: result file not found after job: {result_file}", file=sys.stderr)
        raise optuna.exceptions.TrialPruned()

    data = json.loads(result_file.read_text())
    chrf = data["chrf"]
    print(f"[trial {trial_id}] vocab_size={vocab_size}  dev chrF++={chrf:.4f}")
    return chrf


def main():
    args = parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    storage = f"sqlite:///{STUDY_DB}"
    study = optuna.create_study(
        direction="maximize",
        storage=storage,
        study_name=STUDY_NAME,
        load_if_exists=True,
    )

    already_done = len([t for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = args.n_trials - already_done
    if remaining <= 0:
        print(f"Study already has {already_done} complete trials — nothing to do.")
    else:
        print(f"Starting Optuna BPE search: {remaining} trial(s) remaining "
              f"(target {args.n_trials}, done {already_done})")
        study.optimize(objective, n_trials=remaining)

    print("\n=== Optuna BPE results ===")
    print(f"Best vocab_size : {study.best_params['vocab_size']}")
    print(f"Best dev chrF++ : {study.best_value:.4f}")
    print("\nAll trials:")
    for t in sorted(study.trials, key=lambda x: x.number):
        if t.state == optuna.trial.TrialState.COMPLETE:
            print(f"  trial {t.number:2d}  vocab_size={t.params['vocab_size']:6d}"
                  f"  chrF++={t.value:.4f}")


if __name__ == "__main__":
    main()
