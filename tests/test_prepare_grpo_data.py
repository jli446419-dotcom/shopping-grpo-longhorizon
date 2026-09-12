import unittest

from scripts.prepare_grpo_data import (
    build_verl_record,
    catalog_candidate_ids,
    repair_rows,
)


class PrepareGrpoDataTest(unittest.TestCase):
    def test_repair_is_deterministic_disjoint_and_preserves_positions(self):
        rows = [{"task_id": 1}, {"task_id": 2}, {"task_id": 3}]
        kwargs = {
            "protected_ids": {2, 8},
            "excluded_ids": {1, 2, 3, 8},
            "candidates": list(range(1, 11)),
            "seed": 17,
        }

        first, replacements = repair_rows(rows, **kwargs)
        second, _ = repair_rows(rows, **kwargs)

        self.assertEqual(first, second)
        self.assertEqual(first[0], {"task_id": 1})
        self.assertEqual(first[2], {"task_id": 3})
        self.assertNotIn(first[1]["task_id"], kwargs["excluded_ids"])
        self.assertEqual(replacements[0]["old_task_id"], 2)

    def test_catalog_candidates_require_a_usable_instruction(self):
        catalog = [
            {"instructions": [{"instruction": "buy a mug"}]},
            {"instructions": [{"instruction_simple": "buy a cup"}]},
            {"instructions": []},
            {},
        ]

        self.assertEqual(catalog_candidate_ids(catalog), [0, 1])

    def test_verl_record_keeps_task_identity_in_extra_info(self):
        record = build_verl_record(
            task_id=42, instruction="buy a mug", split="train", index=3
        )

        self.assertEqual(record["data_source"], "shopsimulator")
        self.assertEqual(record["prompt"][-1]["content"], "buy a mug")
        self.assertEqual(record["extra_info"], {"split": "train", "index": 3, "task_id": 42})
        self.assertEqual(record["reward_model"]["style"], "rule")


if __name__ == "__main__":
    unittest.main()
