from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).parents[2]
CONFIG = (
    REPOSITORY_ROOT
    / "ue_project"
    / "ue_framework"
    / "configs"
    / "tausb_sdh_dgcaip_p4_sparse_e20_r3_v1.yaml"
)
WRAPPER = (
    REPOSITORY_ROOT
    / "research_workspace"
    / "experiments"
    / "TAUSB-SDH-DGCAIP-S0-P4-SPARSE-E20"
    / "pre_run"
    / "run_p4_sparse_e20_oneboot_r3.sh"
)
EXPECTED_ROOT = (
    "/root/autodl-tmp/tausb-dgcaip-runs/"
    "TAUSB-SDH-DGCAIP-S0-P4-SPARSE-E20-R3"
)


def test_r3_wrapper_and_config_bind_the_same_fresh_mechanism_root() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    wrapper = WRAPPER.read_text(encoding="utf-8")
    assert config["runtime"]["artifact_root"] == EXPECTED_ROOT
    assert 'exp_root="${data_root}/tausb-dgcaip-runs/' in wrapper
    assert 'TAUSB-SDH-DGCAIP-S0-P4-SPARSE-E20-R3"' in wrapper
    assert "tausb_sdh_dgcaip_p4_sparse_e20_r3_v1.yaml" in wrapper
    assert "TAUSB-SDH-DGCAIP-S0-P4-SPARSE-E20-R2" not in wrapper
    assert 'launch_lock="${exp_root}-LAUNCH-LOCK"' in wrapper
    assert 'if ! mkdir "${launch_lock}"' in wrapper
    assert wrapper.index('if ! mkdir "${launch_lock}"') < wrapper.index(
        "trap shutdown_instance"
    )
    assert 'if [[ -e "${outer_log}" ]]' in wrapper
    assert '>"${outer_log}" 2>&1' in wrapper
