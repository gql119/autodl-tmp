from __future__ import annotations

import importlib
import os
import sys


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


TEST_MODULES = [
    "tests.test_class_routing",
    "tests.test_class_conditioned_loss",
    "tests.test_gradient_extractor",
    "tests.test_virtual_update",
    "tests.test_no_parameter_leak",
    "tests.test_p2_inner_full_loss",
    "tests.test_localized_support",
    "tests.test_supervision_decomposer",
    "tests.test_supervision_interventions",
    "tests.test_loss_reconstruction",
    "tests.test_box_dfl_isolation",
    "tests.test_ambiguous_routing",
    "tests.test_gradient_leakage_diagnostics",
    "tests.test_functional_sgd",
    "tests.test_j3_rollout",
    "tests.test_matched_trajectory_inputs",
    "tests.test_dynamic_assignment_rollout",
    "tests.test_learning_gain",
    "tests.test_learning_gain_objective_sign",
    "tests.test_j3_gradient_to_delta",
    "tests.test_j3_parameter_leak",
    "tests.test_j3_localized_support",
    "tests.test_trajectory_validity_filter",
]


def main() -> None:
    for module_name in TEST_MODULES:
        module = importlib.import_module(module_name)
        for name in sorted(dir(module)):
            if name.startswith("test_"):
                test_fn = getattr(module, name)
                print(f"RUN {module_name}.{name}")
                test_fn()
                print(f"PASS {module_name}.{name}")
    print("ALL_PASS")


if __name__ == "__main__":
    main()
