# Study progress

## Current phase: Experiment design and minimal validation

## Session checkpoint: 2026-09-10

- Paused after completing the Environment/Harness learning phase.
- No environment service, model inference, training, dependency installation,
  or benchmark evaluation has been run.
- Next session starts with the top-level decision tree in
  `environments/ShopSimulator/shop_env/web_agent_site/engine/reward.py`, using
  the white-thermos example to trace `evaluate_purchase()`.

## Roadmap status

| Phase | Status | Remaining focus |
|---|---|---|
| Environment and Harness | Complete | Lightweight integration validation only |
| Reward v3 and termination | Complete | Hand-check against runtime when the environment is available |
| Trajectory curation and SFT | Complete conceptually | Dry-run, data integrity checks, then a minimal smoke run |
| Online GRPO | Complete conceptually | Runtime preflight and a single-update smoke run before training |
| Evaluation and experiments | In progress | Experiment controls, resource plan, orchestration gap, reproducibility |

The historical checkpoints below preserve the order in which the project was
learned. The current overall conceptual progress is approximately 80%.

### Understood

- `src/shopping_grpo/environment/tools.py` defines the model-visible tool schema
  and translates structured tool calls into ShopSimulator actions.
- An ASIN is a product-record identifier. The `12345678` value in
  `examples/trajectories.jsonl` is a sanitized placeholder that connects search,
  opening a candidate, purchasing, and terminal reward evaluation.
- Product identity and selected variant options are separate; a valid purchase
  may require selecting options and checking the resolved variant price.
- The action guard checks whether an operation is legal on the current page;
  Reward v3 evaluates whether a legal terminal decision is good.
- Observation v2 exposes only public evidence. Projection may shorten content
  but must preserve the actionable ASIN and button sets.
- `ShopAgentEnv` owns one `reset -> step -> release` lease. `ShopSimulatorSession`
  binds that lease and its runtime state to one concurrent trajectory.
- The ordinary rollout and veRL AgentLoop are parallel execution paths sharing
  the same environment contract: the former serves collection/evaluation, while
  the latter collects online experience for the veRL trainer.

### Completed

- Read the responsibilities and main control flow of `tools.py`.
- Traced `search_products`, `open_product`, and `buy_now` in the synthetic
  white-thermos example.
- Completed the environment Harness topics: action guarding and correction,
  observation rendering/projection, client leases, concurrent sessions, the
  ordinary rollout loop, and the veRL AgentLoop integration boundary.

### Next

- Read the Reward v3 decision order: compiled requirements, category/budget
  hard gates, weighted preferences, evidence validity, and terminal mapping.
- Hand-calculate outcomes for the running white-thermos example before running
  reward tests.

### Open questions

- None carried forward from the Harness phase. Missing price evidence is
  `reward_unverifiable`; a known over-budget purchase is `wrong_purchase`.

## Session checkpoint — 2026-09-11

### Completed and understood

- Confirmed that Reward v3 conditions are compiled deterministically from the user instruction, structured task annotations, target-product metadata, normalization/matching rules, option-axis mapping, and budget parsing; no LLM judge is used at training or evaluation time.
- Distinguished user requirements from target-product metadata: only explicitly requested and reliably mapped attributes become reward conditions; unrelated target attributes do not.
- Connected terminal Reward v3 scores to GRPO group-relative advantages. A positive raw reward can still produce a negative advantage when it is below the same-task group mean.
- Understood why a same-task group with identical rewards provides no useful relative learning signal.
- Understood incremental `response_mask` construction and alignment: model-generated token spans receive 1, environment/tool observations and padding receive 0; token ids, masks, and old log probabilities must be padded and truncated together.
- Distinguished `attention_mask` from `response_mask`: environment observations remain visible context but do not participate in policy loss.
- Understood the roles of the frozen reference policy, rollout/old policy, and trainable current policy.
- Worked through the probability ratio and clipping objective: clipping limits excessive per-update policy changes but does not forcibly rewrite the actual ratio to the clipping boundary.
- Understood trajectory-level long-horizon credit assignment: one terminal advantage is shared across actor-generated tokens, so correct early actions can receive negative updates in a failed trajectory; useful behavior is learned statistically across many rollouts.
- Identified remaining length effects: masked environment observations do not directly enter policy loss but still consume context and compute; loss aggregation determines whether longer actor trajectories receive more total influence.

### Current conclusions

- SFT is the behavioral warm start that makes GRPO exploration produce sufficiently legal, complete, and comparable shopping trajectories.
- Reward v3's graded terminal outcomes mitigate sparse-reward problems but do not provide true per-step credit assignment.
- Action Guard, observation projection, partial rewards, loop/max-step termination, group-relative comparison, response masking, clipping, and KL regularization address different failure modes and should not be treated as interchangeable safeguards.
- Overall conceptual progress is approximately 45%; the environment/harness and Reward v3 foundations are mostly complete, while dynamic sampling, exact veRL data plumbing, loss aggregation/configuration, training orchestration, evaluation, and experiment design remain.

### Next step

- Start with GRPO dynamic sampling: zero-variance groups, invalid-trajectory group filtering, resampling efficiency, and possible sampling bias.
- Then trace the exact repository code path from `ShoppingToolAgentLoop` output through veRL batch construction, advantage computation, masked policy loss, clipping, and KL handling once source inspection is available.

## Session checkpoint — 2026-09-12

### Completed and understood

- Traced dynamic sampling by `uid`: any sampling-invalid trajectory rejects its whole group, constant-utility groups are removed, and reward-varying groups are retained intact. With two prompts and four rollouts, one optimizer update consumes eight accepted trajectories; repeated insufficient sampling skips the update instead of training on a malformed batch.
- Confirmed the repository's actual GRPO baseline: group-mean centering without standard-deviation normalization, symmetric ratio clipping at 0.8/1.2, token-mean actor loss, no KL term, no entropy bonus, and LoRA rank 16 with alpha 32.
- Traced `ShoppingToolAgentLoop` as the project boundary around veRL's generic `ToolAgentLoop`: the project owns ShopSimulator session lifecycle, observation projection, task identity, reward settlement, and diagnostics; veRL owns generation-loop plumbing, token masks, log probabilities, batch construction, and optimization.
- Distinguished scalar `reward_score` from structured `shopping` diagnostics, and understood how trajectory-level advantages are broadcast across actor response tokens while `response_mask` excludes tool observations and padding from policy loss.
- Understood the remaining token-mean length effect: a longer valid actor response contributes more optimized tokens to the batch even though environment observations remain context-only.
- Studied TRACE as an optional, default-off long-horizon credit experiment. It adds frozen-base, turn-level progress credit to the outcome advantage, but may conflict with Reward v3 when a valid alternative is preferred over the canonical Gold target.
- Traced training orchestration and diagnostics: ShopSimulator runs separately; vLLM generates actions; Ray/veRL coordinate rollout and updates; training uses stochastic groups while validation uses deterministic single rollouts; dynamic filtering occurs before advantage computation.
- Completed the SFT data path: strict teacher-trajectory acceptance, removal of rejected calls and matching Guard messages, assistant-only labels, overlength-example rejection, three-stage curriculum, fresh LoRA per stage, and merge-before-next-stage behavior.
- Distinguished online GRPO handling of Guard rejection from SFT curation: a rejected assistant tool call remains an actor response in a retained online trajectory, while the Guard feedback itself is context-only.
- Traced Baseline/SFT/GRPO evaluation under one deterministic protocol. Strict Gold success is decided by Reward v3 and terminal code, while frozen Rubrics and the Pro Judge provide separate semantic and trajectory-quality panels.
- Understood fixed-denominator and paired comparison rules: infrastructure-invalid and missing Final-200 tasks remain in the 200-task outcome denominator; invalid trajectories are `not_judged`, not zero-scored; Judge deltas use only tasks with valid results on both sides.
- Identified two evaluation documentation/integration issues: `docs/evaluation.md` contains one stale reference to a denominator of 183 although the code and dataset enforce Final-200, and no clear public launcher currently orchestrates the complete TaskFacts/Flash/Pro/four-panel pipeline end to end.

### Current conclusions

- The repository's central design is now connected end to end: deterministic environment and Reward contracts produce auditable terminal outcomes; SFT supplies a legal and useful behavioral prior; GRPO performs group-relative outcome optimization; optional TRACE explores finer long-horizon credit; fixed-protocol evaluation separates outcome, requirements, trajectory quality, and infrastructure.
- Training reward alone cannot establish improvement. Model selection and claims must use fixed validation tasks, Final-200 paired transitions, Reward-type migrations, legality/repetition diagnostics, Judge coverage, and per-dimension trajectory evidence.
- A reduction in trajectory length is beneficial only when conditioned on preserved or improved outcomes; premature wrong purchases can look efficient under an unconditional step-count delta.
- The canonical resource contract is approximately 48 GB for SFT and one 96 GB GPU for GRPO. A 24 GB RTX 4090 is appropriate for checks and deliberately reduced smoke tests, not for claiming reproduction of the canonical GRPO recipe.
- The reported step-100 GRPO result predates the TRACE integration. Git history shows that TRACE was added later as a default-off research extension, so the reproduction control remains outcome-only GRPO; TRACE should be retained as an optional treatment arm, not enabled in the baseline.
- The current curriculum SFT is the only SFT recipe worth running for new work. Historical `data/sft/` and the old three-epoch result are provenance baselines, not a reason to spend GPU time on basic SFT hyperparameter sweeps.

### Next

- Use the shortest defensible base preparation: local static/data checks, one environment/rollout smoke test, one canonical curriculum SFT run, and one deterministic development evaluation. Do not run learning-rate, LoRA-rank, or target-module SFT ablations unless the base policy fails its protocol gate.
- Before TRACE training, run the documented offline proxy audit on existing GRPO diagnostics: test whether frozen-reference canonical-target progress correlates with Reward v3 and whether it systematically penalizes valid alternatives. Only a passing proxy audit advances to matched outcome-GRPO versus GRPO+TRACE training.
- Diagnose the outcome-only GRPO run before choosing further post-training arms. Prioritize group information and rollout diversity; enable Clip-Higher only if entropy falls while upper clipping is active, and enable length shaping only if successful trajectories contain demonstrably redundant steps. Keep KL-off as the control.
- Define run manifests and acceptance gates for reproducibility: code/data/config hashes, seeds, model lineage, environment and Reward versions, GPU/runtime versions, infrastructure-valid rate, and checkpoint-selection rules.
- Resolve the evaluation orchestration gap and stale Final-183 sentence before treating the full four-panel evaluation as a one-command reproducible workflow.

### Open questions and risks

- Actual peak memory and throughput for the canonical GRPO and TRACE runs must be measured on the selected 96 GB GPU; TRACE's repeated prefix scoring may materially increase token volume and wall time.
- AutoDL image compatibility with the repository's pinned Python, PyTorch, CUDA, veRL, vLLM, and Flash-Attention stack must be checked before installing or modifying any environment.
- Full Flash/Pro evaluation requires paid API approval. Strict success and deterministic behavior metrics remain available without the Judge API.
- The GRPO launcher drift was resolved locally: both `scripts/train_grpo.py`
  and `scripts/run_experiment.py` now default to
  `outputs/models/sft-curriculum/stage-c/merged`. Cloud commands still pass the
  path explicitly so the resolved run manifest remains self-explanatory.
- The checked-in datasets are usable but no longer fully task-disjoint after the Pure V4 expansion. `data/sft_pure_v4/all.jsonl` overlaps GRPO train on task IDs 7682, 8993, 11298, 12380, 14953, and 17219; four of these (8993, 11298, 12380, 17219) are SFT gradient rows and two are curriculum development rows. Final-200 remains disjoint. Before new RL training, deterministically replace these six GRPO-train tasks and regenerate the matching JSONL, Parquet, hashes, and metadata; do not recollect SFT teacher trajectories.

## Local preparation checkpoint — 2026-09-12

### Completed locally

- Migrated stale rollout/projection test fixtures from historical `[SEP]`
  pages to observation v2. Production parsing was not weakened with a legacy
  compatibility path.
- Replaced pre-v3 reward-component tests with direct Reward v3 validation and
  terminal-utility tests, including Gold, valid alternative, unverifiable, and
  unfinished trajectories.
- Added `scripts/prepare_grpo_data.py`. Its repair mode deterministically writes
  a new task-disjoint candidate under `outputs/`, while its Parquet mode rebuilds
  the historical veRL schema from live Environment v2.1 reset instructions.
  Both modes refuse existing outputs.
- Extended `scripts/run_experiment.py` with explicit GRPO training-step,
  save/test frequency, and initial-validation controls. This makes one-update
  smoke and 20-update pilot commands auditable without adding another launcher.
- Recorded the stable experiment ladder and AutoDL commands in
  `experiments/README.md`: isolated data → one-update control smoke → 20-update
  control diagnosis → matched treatment pilot → selected 100-update comparison.

### Local experiment results

- Final focused contract/script suite: 54 tests passed, 0 failed (the earlier
  51-test run was extended with three public-entrypoint checks after aligning
  the default GRPO model path).
- Full Python 3.12 suite: 187 discovered; 180 passed, 6 dependency-related
  errors, 1 optional wheel test skipped. Four errors require Hydra to compose
  the veRL config, one requires veRL, and one requires NumPy (and subsequently
  the training stack). No dependency was installed or changed.
- GRPO isolation repair used seed `20260912`, retained 1000 train and 50
  validation tasks, and replaced:
  `7682→14678`, `8993→18888`, `11298→17268`, `12380→16462`,
  `14953→4755`, and `17219→19304`.
- The generated candidate has zero train overlap with Pure V4 SFT, GRPO
  validation, and Final-200. Its train JSONL SHA-256 is
  `9b28f2b8e8eacd4ea35d99ab8bd8acc9a68df0604106df4d33a15b82b7c8d798`;
  the complete audit is in
  `outputs/data-prep/grpo-disjoint/repair-report.json`.
- Dry-run command construction succeeded for outcome-only GRPO at one update
  and TRACE at 20 updates. No environment service, model inference, training,
  checkpoint merge, download, paid API, or Final-200 run was started.

### Remaining gate before cloud training

- On AutoDL, use the already-approved project environment setup, start
  ShopSimulator Environment v2.1, and build the two Parquet files from the
  repaired JSONL. Review task-order metadata and hashes before using them.
- Confirm the merged curriculum Stage C model exists. If it does not, run the
  canonical curriculum once; do not launch basic SFT ablation grids.
- Run the GRPO runtime preflight and then the one-update outcome-only smoke.
  Record peak GPU memory, wall time, environment-valid rate, generated/accepted
  group counts, and whether the first optimizer update completes.
- Parquet creation was not attempted locally because the lightweight Python
  environment lacks `pyarrow`, and installing dependencies requires explicit
  approval. The six full-suite errors likewise reflect the absent cloud
  training stack rather than failures in the locally testable contracts.

## AutoDL environment checkpoint — 2026-09-12

- Provisioned an RTX PRO 6000 Blackwell Server Edition with 94.97 GiB visible
  memory, CUDA-capable driver 595.58.03, 148 GiB data volume, and approximately
  1 TiB host memory.
- Installed the locked main stack successfully: Python 3.12, PyTorch 2.11.0
  with CUDA 13.0, veRL 0.8.0, vLLM 0.25.1, Ray 2.56.1, NumPy 2.2.6, and
  PyArrow 25.0.0. CUDA and BF16 checks passed on the target GPU.
- Built the isolated ShopSimulator Python 3.10.21 environment and the 23,421
  product BM25 index. Product and index hashes matched the repository contract,
  and the veRL dynamic-sampling patch applied successfully.
- The first complete cloud suite discovered 208 tests: 202 passed, 5 failed,
  and 1 optional wheel test skipped. All five failures were stale assertions in
  `tests/test_verl_adapter.py` that were previously hidden locally by the absent
  veRL dependency: they referenced removed `reward_components`, rejected the
  minimized public Reward v3 detail field, supplied pre-v3 terminal fixtures,
  or expected the retired `asin_not_visible` Guard reason. Production code was
  not changed; the fixtures were updated to the current contracts pending a
  cloud rerun.
- The mirror-assisted manual `uv sync` changed `uv.lock`; this is environment
  provenance drift rather than an intended source change. `scripts/setup.sh`
  now uses `uv sync --locked` so future setup attempts fail instead of silently
  rewriting the lock. The cloud checkout must restore its generated lock-file
  edit before pulling the test fix.
- After installation, the data disk used 21 GiB and retained 128 GiB free. No
  model weights, training rollout, optimizer update, or paid evaluation was
  started.
- Cloud rerun at commit `f92e74a` completed all 208 discovered tests in
  20.436 seconds: 207 passed and 1 optional test was skipped. This closes the
  environment-and-contract validation gate; there are no remaining test
  failures before model/runtime preflight.

## Canonical SFT checkpoint — 2026-09-12

- Downloaded the public `Qwen/Qwen3.5-2B` base from ModelScope and completed
  the canonical Pure V4 curriculum through Stage C on one RTX 6000 96 GB GPU.
- Stage A kept 256 train / 28 validation examples, finished in 14.7 minutes,
  and reported train/eval loss 0.4216/0.3528 with 35.79 GiB peak allocation.
- Stage B kept 799 / 88 examples, finished in 51.2 minutes, and reported
  train/eval loss 0.3595/0.3384 with 49.93 GiB peak allocation.
- Stage C kept 1,069 / 118 examples, finished in 76.2 minutes, and reported
  train/eval loss 0.3318/0.3478 with 68.97 GiB peak allocation. Four train and
  one validation trajectory from the manifest exceeded 24,576 tokens and were
  intentionally dropped whole rather than truncated.
- Each stage produced a 277 MiB LoRA adapter and a 4.2 GiB standalone merged
  model. `stage-c/merged` contains readable Qwen3.5 weights and is the sole
  supported GRPO starting policy. Total A-to-C wall time was about 2 h 23 min.
- `train()` stores epoch evaluation metrics in Trainer log history, so the
  terminal `eval_loss=N/A` line is a reporting defect rather than a missing
  validation run. This cosmetic issue remains to be repaired.
- Next gate: run a small non-Final-200 Stage C behavior smoke, then one real
  outcome-only GRPO optimizer update if tool behavior and Reward v3 are valid.

## Stage C behavior and GRPO launcher checkpoint — 2026-09-12

- A deterministic 10-task Stage C smoke completed all tasks: 3 strict
  `gold_purchase`, 5 valid `partial_alternative_purchase`, and 2
  `reward_unverifiable`. Mean final reward was 0.3041, mean trajectory length
  was 8.2 steps, and Reward v3 evidence was valid for 8/10 trajectories.
- The three strict successes selected the target ASIN and every required
  option. The five alternatives retained complete evidence and supplied both
  positive and negative graded outcomes, so the behavior prior is sufficient
  for a one-update outcome-only GRPO engineering smoke.
- Task 643 bought a substitute without selecting its price-determining color
  option. Task 2718 found the target, but a whitespace difference between the
  raw option and the normalized observation footer caused the Guard to reject
  `select_option`; the model then bought without recovering. Both tasks were
  correctly marked `sampling_invalid=true` because variant price was
  unverifiable. The latter also exposes a Guard/projection canonicalization
  risk to quantify before longer training.
- vLLM served successfully on the Blackwell host with
  `VLLM_USE_FLASHINFER_SAMPLER=0`; this bypassed the unavailable FlashInfer
  sampler JIT path without changing the model or project dependencies.
- Raw artifacts are retained under the ignored local directory
  `outputs/cloud-runs/2026-09-12/stage-c-smoke10`. The transfer archive
  SHA-256 is
  `d01ca2702578b6690ba6fe2d49f75e29fb5c33e78f24ce108e927ea871172e79`.
- The first real one-update launch stopped before preflight with
  `NameError: overrides is not defined`; no model loading or optimizer update
  occurred. The launcher now constructs Hydra overrides through one shared
  helper used by both preflight and veRL, and a non-dry-run regression test
  covers the previously untested path.
- The next preflight correctly detected that `data/environment.json` still
  contained the pre-`d99a0ac` hash for `web_agent_text_env.py`. That commit
  intentionally removed an undefined page-rendering argument, changing the
  checked-in file hash from `448f…` to `d6db…`, but did not update the frozen
  manifest. The manifest now identifies the repaired checkout, and a regression
  test verifies every frozen runtime-file hash against the repository.
- Next gate: sync the launcher and runtime-contract fixes to AutoDL, then rerun the same one-update
  outcome-only GRPO smoke. Do not interpret that engineering run as an
  effectiveness result.

## Outcome-only GRPO one-update checkpoint — 2026-09-12

- The first real outcome-only optimizer update completed in 8 minutes 43
  seconds on the RTX PRO 6000. The run reached `training/global_step=1`,
  reported `training/optimizer_updated=1`, and wrote a `global_step_1`
  checkpoint plus three structured diagnostics records.
- Dynamic sampling generated four prompt groups / 16 trajectories over two
  generation batches. Two all-gold groups were correctly dropped as
  `constant_reward`; two reward-varying groups were retained, for an effective
  group ratio of 0.5. No trajectory was infrastructure-invalid,
  reward-unverifiable, sampling-invalid, overlong, aborted, or stuck in a
  repeated-action loop.
- The retained group rewards were
  `[-0.265625, -0.196875, -0.196875, -0.196875]` and
  `[0.04375, 0.04375, 1.0, 0.04375]`. They produced non-zero advantages
  (minimum -0.2391, maximum 0.7172), policy loss -0.02222, gradient norm
  0.1762, and an update at learning rate 1e-6.
- Actor peak allocated/reserved GPU memory was 54.06/73.14 GiB. The trained
  batch contained 49,306 tokens, response lengths ranged from 2,724 to 5,594
  tokens with no clipping, and the measured step throughput was about 600.6
  tokens/second.
- Despite `val_before_train=false` and periodic save/test frequencies greater
  than the run length, veRL performed termination-time validation and checkpoint
  saving. Final validation reward mean@1 was 0.3431, but this has no matched
  pre-update value and therefore is not evidence of one-step improvement.
- A validation DataLoader worker emitted `Killed` during shutdown after
  validation generation ended. The parent trainer still recorded the completed
  update, final validation metrics, and checkpoint, so this run treats it as a
  non-fatal cleanup warning. It remains a stability signal to monitor during
  the control pilot.
- Gate decision: the complete Stage C → rollout → Reward v3 filtering → dynamic
  group selection → GRPO advantage → backward/optimizer → validation/checkpoint
  path is now empirically validated. The next experimental step is a short
  outcome-only control pilot, followed by diagnosis before selecting one
  strategy treatment.
