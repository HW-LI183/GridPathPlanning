"""使用简短可执行示例验证评估分母、漏检及格式错误处理。"""

import pytest

from gridpath.executor import execute
from gridpath.metrics import (
    ParsedVerification,
    aggregate_path_metrics,
    aggregate_repair_metrics,
    aggregate_verification_metrics,
    parse_verification_label,
)
from gridpath.types import ErrorReason, Grid


@pytest.fixture
def grid():
    return Grid(2, 3, frozenset(), (0, 0), (0, 2))


@pytest.mark.parametrize("reason", list(ErrorReason))
def test_verification_parser_normalizes_case_and_whitespace(reason):
    label = f"  INVALID  | STEP = 2 | REASON = {reason.value.upper()} \n"
    assert parse_verification_label(label) == ParsedVerification(False, 2, reason)
    assert parse_verification_label(" \tVaLiD\n") == ParsedVerification(True)


@pytest.mark.parametrize(
    "label",
    [
        "",
        "valid because the goal is reached",
        "invalid",
        "invalid | step=0 | reason=out_of_bounds",
        "invalid | step=-1 | reason=out_of_bounds",
        "invalid | step=1.0 | reason=out_of_bounds",
        "invalid | step=1 | reason=unknown",
        "invalid | step=1 | reason=ınvalid_action_token",
        "invalid | step=1 | reason=out_of_bounds | position=(0,0)",
        "invalid | step=1 | reason=out_of_bounds\nExplanation: left the grid",
        "invalid | reason=out_of_bounds | step=1",
        "valid | step=1 | reason=out_of_bounds",
    ],
)
def test_verification_parser_rejects_explanations_and_incomplete_labels(label):
    assert parse_verification_label(label) is None


def test_path_metrics_distinguish_failure_feasibility_and_successful_detours(grid):
    paths = ["right right", "down right right up", "right", "up"]
    result = aggregate_path_metrics([execute(grid, path) for path in paths], [2] * 4)

    assert result["count"] == 4
    assert result["success_count"] == 2
    assert result["feasible_count"] == 3
    assert result["optimal_success_count"] == 1
    assert result["success_rate"] == 0.5
    assert result["feasibility"] == 0.75
    assert result["optimal_success_rate"] == 0.25
    assert result["conditional_optimal_rate"] == 0.5
    assert result["mean_success_optimality_ratio"] == 0.75


def test_unreachable_references_are_explicitly_excluded_from_optimality(grid):
    unreachable = Grid(2, 2, frozenset({(0, 1), (1, 0)}), (0, 0), (1, 1))
    unreachable_result = execute(unreachable, "")
    result = aggregate_path_metrics([execute(grid, "right right"), unreachable_result], [2, None])

    assert result["count"] == 2
    assert result["success_rate"] == 0.5
    assert result["reference_reachable_count"] == 1
    assert result["reference_unreachable_count"] == 1
    assert result["optimal_success_rate"] == 1.0
    only_unreachable = aggregate_path_metrics([unreachable_result], [None])
    assert only_unreachable["success_rate"] == 0.0
    assert only_unreachable["optimal_success_rate"] is None
    assert only_unreachable["conditional_optimal_rate"] is None
    assert only_unreachable["mean_success_optimality_ratio"] is None


def test_zero_distance_empty_solution_and_successful_loop_have_defined_ratios():
    grid = Grid(1, 2, frozenset(), (0, 0), (0, 0))
    result = aggregate_path_metrics([execute(grid, ""), execute(grid, "right left")], [0, 0])

    assert result["success_rate"] == 1.0
    assert result["optimal_success_rate"] == 0.5
    assert result["conditional_optimal_rate"] == 0.5
    assert result["mean_success_optimality_ratio"] == 0.5


def test_successful_path_cannot_be_marked_unreachable(grid):
    with pytest.raises(ValueError, match="不可达"):
        aggregate_path_metrics([execute(grid, "right right")], [None])


def test_path_metrics_empty_and_no_success_denominators_are_none(grid):
    empty = aggregate_path_metrics([], [])
    assert empty["count"] == 0
    assert empty["success_rate"] is None
    assert empty["optimal_success_rate"] is None
    no_success = aggregate_path_metrics([execute(grid, "right")], [2])
    assert no_success["optimal_success_rate"] == 0.0
    assert no_success["conditional_optimal_rate"] is None
    assert no_success["mean_success_optimality_ratio"] is None


def test_path_metrics_reject_silently_dropped_rows_and_inconsistent_references(grid):
    success = execute(grid, "right right")
    with pytest.raises(ValueError):
        aggregate_path_metrics([success], [])
    with pytest.raises(ValueError):
        aggregate_path_metrics([success], [3])
    with pytest.raises(ValueError):
        aggregate_path_metrics([success], [-1])


def test_verification_misses_and_bad_formats_remain_in_all_denominators():
    truths = [
        "valid",
        "invalid | step=1 | reason=obstacle_collision",
        "invalid | step=2 | reason=out_of_bounds",
        "invalid | step=3 | reason=goal_not_reached",
        "invalid | step=4 | reason=invalid_action_token",
    ]
    predictions = [
        "valid",
        "valid",
        "unparseable model response",
        "invalid | step=3 | reason=obstacle_collision",
        "invalid | step=2 | reason=invalid_action_token",
    ]
    result = aggregate_verification_metrics(truths, predictions)

    assert result["count"] == 5
    assert result["invalid_count"] == 4
    assert result["format_error_count"] == 1
    assert result["format_error_rate"] == 0.2
    assert result["macro_f1"] == pytest.approx(2 / 3)
    assert result["classes"]["valid"]["fp"] == 1
    assert result["classes"]["invalid"]["fp"] == 0
    assert result["classes"]["invalid"]["fn"] == 2
    assert result["first_error_localization_accuracy"] == 0.25
    assert result["error_type_accuracy"] == 0.25
    assert result["joint_diagnosis_accuracy"] == 0.0
    assert result["by_reason"]["goal_not_reached"]["first_error_localization_accuracy"] == 1.0
    assert result["by_reason"]["out_of_bounds"]["first_error_localization_accuracy"] == 0.0


def test_verification_accepts_executor_truths_and_counts_joint_diagnoses(grid):
    truths = [execute(grid, "up"), execute(grid, "right right")]
    result = aggregate_verification_metrics(truths, [truth.verification_label for truth in truths])

    assert result["macro_f1"] == 1.0
    assert result["first_error_localization_accuracy"] == 1.0
    assert result["error_type_accuracy"] == 1.0
    assert result["joint_diagnosis_accuracy"] == 1.0
    assert result["by_reason"]["obstacle_collision"]["joint_diagnosis_accuracy"] is None


def test_verification_fixed_classes_and_empty_denominators():
    valid_only = aggregate_verification_metrics(["valid"], ["valid"])
    assert valid_only["macro_f1"] == 0.5
    assert valid_only["first_error_localization_accuracy"] is None
    bad_format = aggregate_verification_metrics(["valid"], ["valid plus explanation"])
    assert bad_format["macro_f1"] == 0.0
    assert bad_format["classes"]["valid"]["fn"] == 1
    assert bad_format["classes"]["valid"]["fp"] == 0
    empty = aggregate_verification_metrics([], [])
    assert empty["macro_f1"] is None
    assert empty["format_error_rate"] is None
    assert empty["joint_diagnosis_accuracy"] is None


def test_verification_rejects_bad_truths_and_length_mismatches():
    with pytest.raises(ValueError):
        aggregate_verification_metrics(["invalid"], ["valid"])
    with pytest.raises(ValueError):
        aggregate_verification_metrics(["valid"], [])


def test_repair_uses_original_invalid_denominator_and_reports_valid_damage(grid):
    good = execute(grid, "right right")
    bad = execute(grid, "up")
    unfinished = execute(grid, "right")
    result = aggregate_repair_metrics([bad, unfinished, good, good], [good, unfinished, good, bad])

    assert result["original_invalid_count"] == 2
    assert result["repair_success_count"] == 1
    assert result["repair_success_rate"] == 0.5
    assert result["original_valid_count"] == 2
    assert result["valid_destruction_count"] == 1
    assert result["valid_destruction_rate"] == 0.5
    assert result["final_success_rate"] == 0.5


def test_repair_undefined_denominators_and_length_mismatch(grid):
    good = execute(grid, "right right")
    bad = execute(grid, "up")
    assert aggregate_repair_metrics([good], [good])["repair_success_rate"] is None
    assert aggregate_repair_metrics([bad], [bad])["valid_destruction_rate"] is None
    empty = aggregate_repair_metrics([], [])
    assert empty["final_success_rate"] is None
    assert empty["repair_success_rate"] is None
    with pytest.raises(ValueError):
        aggregate_repair_metrics([good], [])
