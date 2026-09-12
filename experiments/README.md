# Experiments

This directory contains the compact, reviewable artifacts behind the README
result table. Large checkpoints and complete trajectories are intentionally not
stored in Git.

```text
baseline/   base-model evaluation config and summary
sft/        SFT training/evaluation config and summary
grpo/       GRPO training/evaluation config and summary
reproduction/ compact metrics from the current study; raw artifacts stay ignored
comparison.md
```

All reported models use the same 200 held-out tasks, Environment v2.1 and
Reward v3. See [comparison.md](comparison.md) for interpretation and protocol
limitations.

## Post-training study protocol

The study spends the minimum GPU time needed to establish a working base, then
uses matched GRPO arms to study post-training strategy. Runtime outputs belong
under `outputs/`; this directory stores only stable protocols and compact final
summaries.

### 1. Prepare isolated GRPO tasks

The checked-in Pure V4 SFT pool overlaps six rows in the historical GRPO train
split. Produce a reviewable replacement candidate without overwriting canonical
data:

```bash
PYTHONPATH=src python scripts/prepare_grpo_data.py repair \
  --output-dir outputs/data-prep/grpo-disjoint
```

The command writes two JSONL files and `repair-report.json`. Advance only when
all `remaining_overlaps` lists are empty. In an environment that already has
the project's `pyarrow` dependency and a running Environment v2.1 service,
build matching veRL files:

```bash
PYTHONPATH=src python scripts/prepare_grpo_data.py build-parquet \
  --tasks outputs/data-prep/grpo-disjoint/train.jsonl \
  --output outputs/data-prep/grpo-disjoint/train.parquet \
  --split train

PYTHONPATH=src python scripts/prepare_grpo_data.py build-parquet \
  --tasks outputs/data-prep/grpo-disjoint/validation.jsonl \
  --output outputs/data-prep/grpo-disjoint/validation.parquet \
  --split validation
```

Both operations refuse to overwrite an existing output. Review their metadata
hashes before training; promotion into `data/grpo/` is a separate decision.

### 2. Establish the control cheaply

Use the merged curriculum Stage C model explicitly. A one-update smoke run
checks the entire ShopSimulator → vLLM → AgentLoop → Reward v3 → optimizer path;
it intentionally skips the initial validation. The pinned veRL trainer still
performs final validation and writes a final checkpoint when the one-step run
terminates, independently of the periodic save/test frequencies:

```bash
PYTHONPATH=src python scripts/run_experiment.py grpo_baseline \
  --model outputs/models/sft-curriculum/stage-c/merged \
  --train-data outputs/data-prep/grpo-disjoint/train.parquet \
  --validation-data outputs/data-prep/grpo-disjoint/validation.parquet \
  --output-root outputs/post-training/smoke \
  --training-steps 1 --save-freq 1000 --test-freq 1000 \
  --skip-val-before-train --dry-run
```

First run with `--dry-run`; remove only that flag after reviewing the resolved
command. The smoke gate is: one optimizer update completes, reward settlement
is valid, response masks/log probabilities align, and no lease or OOM failure
occurs. It is not an effectiveness result.

Next run an outcome-only 20-update pilot with validation at updates 10 and 20:

```bash
PYTHONPATH=src python scripts/run_experiment.py grpo_baseline \
  --model outputs/models/sft-curriculum/stage-c/merged \
  --train-data outputs/data-prep/grpo-disjoint/train.parquet \
  --validation-data outputs/data-prep/grpo-disjoint/validation.parquet \
  --output-root outputs/post-training/pilot-control \
  --training-steps 20 --save-freq 10 --test-freq 10 --dry-run
```

### 3. Test strategy, not basic SFT tuning

Do not run SFT learning-rate, rank, or target-module grids. Run the canonical
curriculum once if the Stage C model is unavailable. Diagnose the control pilot
before choosing an intervention:

- TRACE is the first planned treatment only after its offline proxy audit does
  not systematically penalize valid alternatives.
- Clip-Higher is justified only when entropy collapses while upper clipping is
  active.
- Length shaping is justified only when successful trajectories contain
  redundant steps; shorter failed trajectories are not evidence.
- KL remains off in the control and is enabled only for observed policy drift
  or validation regression.

Run a matched TRACE pilot with the same model, data, seed-level configuration,
and budget:

```bash
PYTHONPATH=src python scripts/run_experiment.py grpo_trace \
  --model outputs/models/sft-curriculum/stage-c/merged \
  --train-data outputs/data-prep/grpo-disjoint/train.parquet \
  --validation-data outputs/data-prep/grpo-disjoint/validation.parquet \
  --output-root outputs/post-training/pilot-trace \
  --training-steps 20 --save-freq 10 --test-freq 10 --dry-run
```

Only arms that pass the pilot gate advance to a 100-update matched comparison
(`--training-steps 100 --save-freq 50 --test-freq 50`). Select checkpoints on
the fixed development validation tasks. Run Final-200 only after selection, and
compare paired task transitions, Reward v3 types, infrastructure-valid rate,
Guard/repetition diagnostics, and success-conditioned trajectory length.
