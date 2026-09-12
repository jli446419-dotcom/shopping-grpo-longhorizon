#!/usr/bin/env python3
"""Prepare disjoint GRPO task splits and their veRL parquet files safely."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from shopping_grpo.environment.client import ShopAgentEnv
from shopping_grpo.evaluation.artifacts import (
    iter_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from shopping_grpo.evaluation.rollout import SYSTEM_PROMPT


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT / "data/grpo/train.jsonl"
DEFAULT_VALIDATION = ROOT / "data/grpo/validation.jsonl"
DEFAULT_SFT = ROOT / "data/sft_pure_v4/all.jsonl"
DEFAULT_EVALUATION = ROOT / "data/evaluation/tasks.jsonl"
DEFAULT_CATALOG = (
    ROOT
    / "environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz"
)


def task_ids(rows: Iterable[Mapping]) -> list[int]:
    ids = [int(row["task_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("task split contains duplicate task_id values")
    return ids


def catalog_candidate_ids(catalog: Sequence[object]) -> list[int]:
    candidates = []
    for task_id, item in enumerate(catalog):
        if not isinstance(item, Mapping):
            continue
        instructions = item.get("instructions")
        if not isinstance(instructions, list):
            continue
        usable = any(
            isinstance(entry, Mapping)
            and bool(entry.get("instruction") or entry.get("instruction_simple"))
            for entry in instructions
        )
        if usable:
            candidates.append(task_id)
    return candidates


def repair_rows(
    train_rows: Sequence[Mapping],
    *,
    protected_ids: set[int],
    excluded_ids: set[int],
    candidates: Sequence[int],
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Replace protected train tasks deterministically while preserving row order."""
    original_ids = task_ids(train_rows)
    overlap = set(original_ids) & set(protected_ids)
    available = sorted(set(map(int, candidates)) - set(excluded_ids))
    random.Random(seed).shuffle(available)
    if len(available) < len(overlap):
        raise ValueError(
            f"not enough eligible catalog tasks: need {len(overlap)}, have {len(available)}"
        )
    replacement_iter = iter(available)
    repaired = []
    replacements = []
    for position, row in enumerate(train_rows):
        updated = dict(row)
        old_id = int(updated["task_id"])
        if old_id in overlap:
            new_id = next(replacement_iter)
            updated["task_id"] = new_id
            replacements.append(
                {"position": position, "old_task_id": old_id, "new_task_id": new_id}
            )
        repaired.append(updated)
    repaired_ids = task_ids(repaired)
    if set(repaired_ids) & set(protected_ids):
        raise AssertionError("repaired GRPO train still overlaps protected tasks")
    return repaired, replacements


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_verl_record(*, task_id: int, instruction: str, split: str, index: int) -> dict:
    return {
        "data_source": "shopsimulator",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(instruction)},
        ],
        "ability": "shopping",
        "reward_model": {"style": "rule", "ground_truth": None},
        "extra_info": {"split": split, "index": index, "task_id": int(task_id)},
    }


def repair_command(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"output directory must not exist: {output_dir}")
    train_rows = list(iter_jsonl(args.train))
    validation_rows = list(iter_jsonl(args.validation))
    sft_ids = set(task_ids(iter_jsonl(args.sft)))
    validation_ids = set(task_ids(validation_rows))
    evaluation_ids = set(task_ids(iter_jsonl(args.evaluation)))
    train_ids = set(task_ids(train_rows))
    with gzip.open(args.catalog, "rt", encoding="utf-8") as stream:
        catalog = json.load(stream)
    if not isinstance(catalog, list):
        raise ValueError("ShopSimulator catalog root must be a list")

    protected_ids = sft_ids | validation_ids | evaluation_ids
    repaired, replacements = repair_rows(
        train_rows,
        protected_ids=protected_ids,
        excluded_ids=train_ids | protected_ids,
        candidates=catalog_candidate_ids(catalog),
        seed=args.seed,
    )
    train_output = output_dir / "train.jsonl"
    validation_output = output_dir / "validation.jsonl"
    write_jsonl_atomic(train_output, repaired)
    write_jsonl_atomic(validation_output, validation_rows)
    report = {
        "contract": "GRPO task isolation from Pure V4 SFT, GRPO validation, and Final-200",
        "seed": args.seed,
        "counts": {"train": len(repaired), "validation": len(validation_rows)},
        "replacements": replacements,
        "source_hashes": {
            "train_jsonl": sha256(args.train),
            "validation_jsonl": sha256(args.validation),
            "sft_jsonl": sha256(args.sft),
            "evaluation_jsonl": sha256(args.evaluation),
            "catalog": sha256(args.catalog),
        },
        "output_hashes": {
            "train_jsonl": sha256(train_output),
            "validation_jsonl": sha256(validation_output),
        },
        "remaining_overlaps": {
            "train_sft": sorted(set(task_ids(repaired)) & sft_ids),
            "train_validation": sorted(set(task_ids(repaired)) & validation_ids),
            "train_evaluation": sorted(set(task_ids(repaired)) & evaluation_ids),
        },
    }
    write_json_atomic(output_dir / "repair-report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parquet_command(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    metadata_output = output.with_suffix(output.suffix + ".metadata.json")
    if output.exists() or metadata_output.exists():
        raise SystemExit(
            f"parquet output or metadata already exists: {output}, {metadata_output}"
        )
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "building parquet requires the project training environment with pyarrow"
        ) from exc

    rows = []
    ids = task_ids(iter_jsonl(args.tasks))
    for index, task_id in enumerate(ids):
        with ShopAgentEnv(base_url=args.env_url, timeout=args.timeout) as env:
            initial = env.reset(task_id)
        instruction = initial.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"task {task_id} reset returned no instruction")
        rows.append(
            build_verl_record(
                task_id=task_id,
                instruction=instruction,
                split=args.split,
                index=index,
            )
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output)
    written_ids = [int(row["extra_info"]["task_id"]) for row in rows]
    if written_ids != ids:
        raise AssertionError("parquet task order differs from source JSONL")
    metadata = {
        "source_jsonl": str(args.tasks.resolve()),
        "source_sha256": sha256(args.tasks),
        "parquet_sha256": sha256(output),
        "split": args.split,
        "rows": len(rows),
        "task_ids_match_source_order": True,
        "environment_version": "shopsimulator-environment-v2.1",
    }
    write_json_atomic(metadata_output, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    repair = subparsers.add_parser("repair", help="write a new disjoint JSONL split")
    repair.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    repair.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    repair.add_argument("--sft", type=Path, default=DEFAULT_SFT)
    repair.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    repair.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    repair.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/data-prep/grpo-disjoint"
    )
    repair.add_argument("--seed", type=int, default=20260912)
    repair.set_defaults(handler=repair_command)

    parquet = subparsers.add_parser(
        "build-parquet", help="query ShopSimulator and write one veRL parquet"
    )
    parquet.add_argument("--tasks", type=Path, required=True)
    parquet.add_argument("--output", type=Path, required=True)
    parquet.add_argument("--split", choices=("train", "validation"), required=True)
    parquet.add_argument("--env-url", default="http://127.0.0.1:5700")
    parquet.add_argument("--timeout", type=int, default=60)
    parquet.set_defaults(handler=parquet_command)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
