#!/usr/bin/env python3
"""Summarize matched Base, SFT, and GRPO evaluation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = (
    "strict_success_rate",
    "purchase_success_rate",
    "mean_final_reward",
    "average_steps",
)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_run(label: str, run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    trajectories_path = run_dir / "trajectories.jsonl"
    if not summary_path.is_file() or not trajectories_path.is_file():
        raise ValueError(f"{label}: missing summary.json or trajectories.jsonl in {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = _read_jsonl(trajectories_path)
    task_ids = [int(row["task_id"]) for row in rows]
    expected = int(summary.get("expected_tasks", len(task_ids)))
    if len(task_ids) != expected or len(set(task_ids)) != expected:
        raise ValueError(
            f"{label}: expected {expected} unique completed tasks, got {len(task_ids)} rows"
        )
    guards = int(
        (summary.get("context_projection") or {}).get(
            "guard_rejections",
            sum(int(value) for value in (summary.get("guard_reason_counts") or {}).values()),
        )
    )
    return {
        "label": label,
        "run_dir": str(run_dir.resolve()),
        "task_ids": task_ids,
        "task_count": expected,
        "strict_success_rate": float(summary["strict_success_rate"]),
        "purchase_success_rate": float(summary["purchase_success_rate"]),
        "mean_final_reward": float(summary["mean_final_reward"]),
        "average_steps": float(summary["average_steps"]),
        "guard_rejections": guards,
        "guard_rejections_per_task": guards / expected if expected else 0.0,
        "guard_reason_counts": summary.get("guard_reason_counts") or {},
        "protocol": summary.get("protocol") or {},
    }


def build_comparison(run_specs: list[str]) -> dict:
    runs = []
    for spec in run_specs:
        if "=" not in spec:
            raise ValueError(f"run must use LABEL=DIR syntax: {spec!r}")
        label, raw_dir = spec.split("=", 1)
        runs.append(_load_run(label.strip(), Path(raw_dir)))
    if len(runs) < 2:
        raise ValueError("at least two runs are required")
    expected_ids = runs[0]["task_ids"]
    for run in runs[1:]:
        if run["task_ids"] != expected_ids:
            raise ValueError(
                f"task order mismatch between {runs[0]['label']} and {run['label']}"
            )
    reference = runs[0]
    for run in runs:
        run["delta_vs_reference"] = {
            metric: run[metric] - reference[metric] for metric in METRICS
        }
        run["guard_delta_vs_reference"] = (
            run["guard_rejections"] - reference["guard_rejections"]
        )
    return {
        "reference": reference["label"],
        "task_count": len(expected_ids),
        "task_ids": expected_ids,
        "runs": runs,
    }


def render_markdown(comparison: dict) -> str:
    lines = [
        "# Base / SFT / GRPO matched evaluation",
        "",
        f"Frozen tasks: {comparison['task_count']}; reference: {comparison['reference']}.",
        "",
        "| Model | Strict success | Purchase success | Mean Reward | Average steps | Guard rejections | Guard/task |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run in comparison["runs"]:
        lines.append(
            "| {label} | {strict:.2%} | {purchase:.2%} | {reward:.4f} | "
            "{steps:.2f} | {guards} | {guard_rate:.3f} |".format(
                label=run["label"],
                strict=run["strict_success_rate"],
                purchase=run["purchase_success_rate"],
                reward=run["mean_final_reward"],
                steps=run["average_steps"],
                guards=run["guard_rejections"],
                guard_rate=run["guard_rejections_per_task"],
            )
        )
    lines.extend(["", "Guard rejection reasons:", ""])
    for run in comparison["runs"]:
        reasons = run["guard_reason_counts"]
        rendered = ", ".join(f"`{key}`={value}" for key, value in sorted(reasons.items()))
        lines.append(f"- {run['label']}: {rendered or 'none'}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="LABEL=RUN_DIR")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = build_comparison(args.run)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.markdown.write_text(render_markdown(comparison), encoding="utf-8")
    print(render_markdown(comparison), end="")


if __name__ == "__main__":
    main()
