# Repository contract

This repository supports one workflow only:

```text
Baseline → SFT → GRPO → Evaluation
```

The runtime contract is ShopSimulator Environment v2.1, Reward v3, observation
v2 and tool schema v2. Do not add compatibility launchers, historical datasets,
old benchmarks, machine-specific paths or experiment journals.

Training data must never overlap `data/evaluation/tasks.jsonl`. Strict success
requires a complete `gold_purchase` terminal result with `reward_valid=true`.

Do not start training, merge models or run the 200-task evaluation unless the
user explicitly requests execution.

## Collaboration and safety

- Read every applicable `AGENTS.md` before operating in the repository.
- Work for this study belongs on the `study/framework-strategy` branch. Keep
  this repository independent from other coursework repositories.
- Reading, searching, editing project source/configuration/documentation, and
  running static checks, unit tests, or lightweight smoke tests are allowed.
- Explain important design goals and trade-offs before making changes. Prefer
  the existing project structure and keep changes small, readable, and focused.
- Ask for explicit approval before installing, removing, or upgrading
  dependencies; changing Python, PyTorch, CUDA, veRL, or vLLM environments;
  downloading models, datasets, or large resources; starting long-running or
  paid GPU jobs; calling paid APIs or operating external accounts; changing
  credentials or environment variables; or running `git commit`, `push`,
  `merge`, `rebase`, or history-rewriting operations.
- Never write API keys, cookies, access tokens, `.env` files, or other
  credentials into Git. Never commit model weights, datasets, checkpoints,
  rollout artifacts, or large logs.
- Never bulk-delete files or directories. Any deletion requires explicit user
  approval and must target one exact file at a time; do not use recursive delete
  commands.
- Do not create scattered teaching/debug scripts. Add a small experiment only
  when it materially improves understanding.

## Study workflow

- Teach module responsibilities, design rationale, data flow, and the problem
  solved by each strategy; do not default to line-by-line Python explanation.
- Prioritize environment, agent, trajectory/rollout, Reward v3, GRPO,
  long-horizon credit assignment, and the relationship between training and
  evaluation. Use one concrete shopping task as a running example for complex
  flows.
- At the start of each learning phase, state its goals. At the end, summarize
  conclusions, validations, unresolved issues, and the next step.
- Validate data flow, environment interaction, and reward calculation before
  proposing full training.
- Maintain one concise learning-progress document at `docs/study-progress.md`.
  Record understood/completed topics, small validation conclusions, unresolved
  questions, and next steps. This is a study index, not an experiment journal.
- After modifications, report changed files, reasons, checks run and their
  results, plus any unresolved issues.
