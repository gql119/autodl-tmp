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
