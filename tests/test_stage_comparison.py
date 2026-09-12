import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_stage_comparison import build_comparison, render_markdown


class StageComparisonTest(unittest.TestCase):
    def _write_run(self, root: Path, name: str, reward: float, guards: int) -> Path:
        run_dir = root / name
        run_dir.mkdir()
        summary = {
            "expected_tasks": 2,
            "strict_success_rate": 0.5,
            "purchase_success_rate": 0.5,
            "mean_final_reward": reward,
            "average_steps": 4.5,
            "guard_reason_counts": {"bad_click": guards},
            "context_projection": {"guard_rejections": guards},
            "protocol": {"temperature": 0.0},
        }
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        rows = [{"task_id": 1}, {"task_id": 2}]
        (run_dir / "trajectories.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return run_dir

    def test_builds_matched_comparison_and_guard_rate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = self._write_run(root, "base", 0.1, 2)
            grpo = self._write_run(root, "grpo", 0.3, 0)
            result = build_comparison([f"Base={base}", f"GRPO={grpo}"])
            self.assertEqual(result["task_count"], 2)
            self.assertEqual(result["runs"][0]["guard_rejections_per_task"], 1.0)
            self.assertAlmostEqual(
                result["runs"][1]["delta_vs_reference"]["mean_final_reward"], 0.2
            )
            self.assertIn("| GRPO | 50.00%", render_markdown(result))

    def test_rejects_task_order_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = self._write_run(root, "base", 0.1, 0)
            grpo = self._write_run(root, "grpo", 0.2, 0)
            (grpo / "trajectories.jsonl").write_text(
                '{"task_id": 2}\n{"task_id": 1}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "task order mismatch"):
                build_comparison([f"Base={base}", f"GRPO={grpo}"])


if __name__ == "__main__":
    unittest.main()
