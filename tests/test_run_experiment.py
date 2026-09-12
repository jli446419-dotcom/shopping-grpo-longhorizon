import unittest

from scripts.run_experiment import ROOT, build_experiment, load_registry, resolve_experiment


class RunExperimentTest(unittest.TestCase):
    def test_grpo_runtime_budget_is_explicit_in_child_command(self):
        registry = load_registry(ROOT / "configs/experiments.json")
        experiment = resolve_experiment(registry, "grpo_baseline")

        command, _, _ = build_experiment(
            experiment,
            root=ROOT,
            training_steps=1,
            save_freq=1000,
            test_freq=1000,
            val_before_train=False,
        )

        self.assertIn("trainer.total_training_steps=1", command)
        self.assertIn("trainer.save_freq=1000", command)
        self.assertIn("trainer.test_freq=1000", command)
        self.assertIn("trainer.val_before_train=false", command)
        self.assertIn("shopping_trace.enable=false", command)

    def test_trace_arm_changes_only_registered_trace_setting(self):
        registry = load_registry(ROOT / "configs/experiments.json")
        experiment = resolve_experiment(registry, "grpo_trace")

        command, _, _ = build_experiment(experiment, root=ROOT, training_steps=20)

        self.assertIn("shopping_trace.enable=true", command)
        self.assertIn("trainer.total_training_steps=20", command)

    def test_nonpositive_runtime_budget_is_rejected(self):
        registry = load_registry(ROOT / "configs/experiments.json")
        experiment = resolve_experiment(registry, "grpo_baseline")

        with self.assertRaisesRegex(ValueError, "must be positive"):
            build_experiment(experiment, root=ROOT, training_steps=0)


if __name__ == "__main__":
    unittest.main()
