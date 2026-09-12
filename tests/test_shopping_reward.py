"""Shopping GRPO 终局奖励的纯函数测试。"""

import unittest

from shopping_grpo.training.grpo.adapter.runtime import (
    make_runtime_state,
    record_action_attempt,
    reward_breakdown,
    validate_reward,
)


def reward_detail(
    *,
    reward_type="gold_purchase",
    reward_valid=True,
    terminal_utility=1.0,
    weighted_score=1.0,
):
    return {
        "reward_version": "shopsimulator-reward-v3",
        "reward_type": reward_type,
        "reward_valid": reward_valid,
        "termination_reason": reward_type,
        "target_asin_match": reward_type == "gold_purchase",
        "hard_gates": {
            "category": {
                "status": "pass",
                "passed": True,
                "verifiable": True,
                "comparator": "category_match",
                "source_field": "category",
            },
            "budget": {
                "status": "pass",
                "passed": True,
                "verifiable": True,
                "comparator": "less_than_or_equal",
                "source_field": "selected_price",
            },
        },
        "weighted_score": weighted_score,
        "evidence_coverage": 1.0,
        "dimension_scores": {
            "brand": 1.0,
            "model": 1.0,
            "core_functions": 1.0,
            "key_options": 1.0,
        },
        "terminal_utility": terminal_utility,
        "purchase_success": reward_type
        in {"gold_purchase", "valid_alternative_purchase"},
        "sampling_invalid": not reward_valid,
    }


def terminal_state(*, reward_type="gold_purchase", native_reward=1.0):
    state = make_runtime_state(task_id=1, max_steps=35)
    detail = validate_reward(
        reward_detail(reward_type=reward_type, terminal_utility=native_reward)
    )
    state.update(
        {
            "done": True,
            "terminal_result": {"done": True, "over": True},
            "final_reward": native_reward,
            "reward_version": detail["reward_version"],
            "reward_type": detail["reward_type"],
            "reward_valid": detail["reward_valid"],
            "reward_detail": detail,
        }
    )
    return state


class ShoppingRewardTest(unittest.TestCase):
    def test_gold_purchase_uses_reward_v3_terminal_utility(self):
        result = reward_breakdown(terminal_state(native_reward=1.0))

        self.assertAlmostEqual(result["full"], 1.0)
        self.assertAlmostEqual(result["strict"], 1.0)
        self.assertAlmostEqual(result["semantic"], 1.0)
        self.assertAlmostEqual(result["efficiency"], 0.0)
        self.assertAlmostEqual(result["total"], 1.0)

    def test_valid_alternative_is_success_but_not_strict_gold(self):
        result = reward_breakdown(
            terminal_state(
                reward_type="valid_alternative_purchase", native_reward=0.8
            )
        )

        self.assertEqual(result["purchase_success"], 1.0)
        self.assertEqual(result["full"], 0.0)
        self.assertEqual(result["strict"], 0.0)
        self.assertEqual(result["total"], 0.8)

    def test_unfinished_trajectory_is_sampling_invalid_without_shaped_penalty(self):
        state = make_runtime_state(task_id=1, max_steps=35)
        state["termination_reason"] = "assistant_finished_without_environment_done"
        state["error"] = state["termination_reason"]

        result = reward_breakdown(state)

        self.assertEqual(result["semantic"], 0.0)
        self.assertEqual(result["penalty_unfinished"], 0.0)
        self.assertEqual(result["total"], 0.0)
        self.assertTrue(result["sampling_invalid"])

    def test_reward_v3_detail_must_be_consistent_finite_and_bounded(self):
        detail = reward_detail()
        detail["termination_reason"] = "wrong_purchase"
        with self.assertRaisesRegex(ValueError, "termination_reason"):
            validate_reward(detail)

        detail = reward_detail()
        detail["terminal_utility"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_reward(detail)

        detail = reward_detail()
        detail["weighted_score"] = 1.1
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            validate_reward(detail)

    def test_validate_reward_minimizes_public_fields(self):
        raw = reward_detail(weighted_score=0.6)
        raw["private_debug"] = "must not survive"

        result = validate_reward(raw)

        self.assertNotIn("private_debug", result)
        self.assertEqual(result["weighted_score"], 0.6)
        self.assertEqual(result["hard_gates"]["category"]["status"], "pass")

    def test_reward_unverifiable_produces_no_learning_signal(self):
        detail = validate_reward(
            reward_detail(
                reward_type="reward_unverifiable",
                reward_valid=False,
                terminal_utility=0.0,
                weighted_score=0.0,
            )
        )
        state = make_runtime_state(task_id=1, max_steps=35)
        state.update(
            {
                "done": True,
                "terminal_result": {"done": True, "over": True},
                "reward_version": detail["reward_version"],
                "reward_type": detail["reward_type"],
                "reward_valid": detail["reward_valid"],
                "reward_detail": detail,
            }
        )

        result = reward_breakdown(state)

        self.assertTrue(result["sampling_invalid"])
        self.assertTrue(result["reward_unverifiable"])
        self.assertEqual(result["total"], 0.0)

    def test_same_action_on_same_page_within_three_attempts_is_repeated(self):
        state = make_runtime_state(task_id=1, max_steps=35)

        record_action_attempt(state, "search_products", {"query": "mug"}, "search page")
        record_action_attempt(state, "open_product", {"asin": "123"}, "search page")
        record_action_attempt(state, "search_products", {"query": "mug"}, "search page")

        self.assertEqual(state["action_attempt_count"], 3)
        self.assertEqual(state["repeat_action_count"], 1)
        self.assertAlmostEqual(reward_breakdown(state)["repeat_action_rate"], 1 / 3)

    def test_different_parameters_or_page_are_not_repeated(self):
        state = make_runtime_state(task_id=1, max_steps=35)

        record_action_attempt(state, "search_products", {"query": "mug"}, "page 1")
        record_action_attempt(state, "search_products", {"query": "cup"}, "page 1")
        record_action_attempt(state, "search_products", {"query": "mug"}, "page 2")

        self.assertEqual(state["repeat_action_count"], 0)

    def test_think_is_not_an_environment_action_attempt(self):
        state = make_runtime_state(task_id=1, max_steps=35)

        record_action_attempt(state, "think", {"note": "plan"}, "page")

        self.assertEqual(state["action_attempt_count"], 0)
        self.assertEqual(state["recent_action_signatures"], [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
