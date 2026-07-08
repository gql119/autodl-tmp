from .diagnostics import GradientLeakageResult, compute_gradient_leakage_matrix
from .early_stopping import EarlyStoppingResult, HeldoutEarlyStopping
from .feasibility import DualState, constrained_objective, constraint_violations, natural_variation_thresholds, raw_counterfactual_gap, summarize_clean_gains, pearson_correlation, spearman_correlation
from .functional_optimizer import clone_parameter_dict, functional_sgd_step, init_functional_sgd_state
from .gain_scale import GainScales, compute_gain_scales_from_rows, robust_min_from_positive_gains, robust_scale_from_clean_gains
from .gradient_diagnostics import gradient_conflict_diagnostics
from .learning_gain import LearningGainMetrics, compute_learning_gain_objective, compute_learning_gain_objective_v2
from .method import J3LearningGainMethod
from .online_sampler import OnlineTrajectorySampler, OnlineTrajectorySample
from .rollout_engine import J3RolloutEngine, RolloutOutput
from .trajectory_sampler import TrajectorySampler
from .trajectory_state import BatchData, FunctionalOptimizerState, RolloutState, TrajectoryBatchSequence

__all__ = [
    "BatchData",
    "EarlyStoppingResult",
    "DualState",
    "FunctionalOptimizerState",
    "GainScales",
    "GradientLeakageResult",
    "HeldoutEarlyStopping",
    "J3LearningGainMethod",
    "J3RolloutEngine",
    "LearningGainMetrics",
    "OnlineTrajectorySample",
    "OnlineTrajectorySampler",
    "RolloutOutput",
    "RolloutState",
    "TrajectoryBatchSequence",
    "TrajectorySampler",
    "clone_parameter_dict",
    "constrained_objective",
    "constraint_violations",
    "compute_gain_scales_from_rows",
    "compute_gradient_leakage_matrix",
    "compute_learning_gain_objective",
    "compute_learning_gain_objective_v2",
    "functional_sgd_step",
    "gradient_conflict_diagnostics",
    "init_functional_sgd_state",
    "natural_variation_thresholds",
    "pearson_correlation",
    "raw_counterfactual_gap",
    "robust_min_from_positive_gains",
    "robust_scale_from_clean_gains",
    "spearman_correlation",
    "summarize_clean_gains",
]
