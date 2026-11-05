"""明确记录分母的离线指标，不调用规划器或真实标签生成器。

参考距离为 ``None`` 表示目标不可达。这些样本保留在成功率与可行率的分母中，
从最优性指标中排除，并记录排除数量。模型预测缺失或格式错误时，仍保留在
评估分母中。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from .types import ErrorReason

if TYPE_CHECKING:
    from .executor import ExecutionResult


@dataclass(frozen=True, slots=True)
class ParsedVerification:
    """解析后的验证标签；无效标签必须包含错误诊断。"""

    valid: bool
    step: int | None = None
    reason: ErrorReason | None = None


_INVALID_LABEL = re.compile(
    r"invalid\s*\|\s*step\s*=\s*([0-9]+)\s*\|\s*reason\s*=\s*("
    + "|".join(re.escape(reason.value) for reason in ErrorReason)
    + r")",
)


def parse_verification_label(text: str) -> ParsedVerification | None:
    """只解析完整的 ``valid`` 或含三个字段的无效标签。

    容许空白和大小写变化，步骤必须为正整数。解释文字、多余字段、带符号的步骤
    以及未知错误类型均无法解析。返回 ``None`` 表示格式错误，不代表任何有效性类别。
    """

    if not isinstance(text, str):
        return None
    label = text.strip().lower()
    if label == "valid":
        return ParsedVerification(valid=True)
    match = _INVALID_LABEL.fullmatch(label)
    if match is None:
        return None
    try:
        step = int(match.group(1))
    except ValueError:
        # 极长数字可能超过 Python 的整数转换限制。
        return None
    if step < 1:
        return None
    return ParsedVerification(False, step, ErrorReason(match.group(2)))


def _rate(numerator: int | float, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def aggregate_path_metrics(
    results: Iterable[ExecutionResult],
    reference_distances: Iterable[int | None],
) -> dict[str, int | float | None]:
    """使用外部提供的离线最短距离评估路径。

    路径长度采用执行器完整序列语义下的 ``executed_steps``。有可达参考的失败样本
    在最优成功率中计零。条件最优率和平均 ``d*/L`` 仅以有可达参考的成功路径为分母。
    长度和参考距离均为零的成功路径，其最优性比取 1。距离或长度不一致时抛出
    ``ValueError``。
    """

    count = success_count = feasible_count = reachable_count = 0
    success_with_reference_count = optimal_success_count = 0
    optimality_ratio_sum = 0.0
    for result, distance in zip(results, reference_distances, strict=True):
        if distance is not None and (type(distance) is not int or distance < 0):
            raise ValueError("参考距离必须为非负整数或 None")
        count += 1
        success_count += bool(result.success)
        feasible_count += bool(result.feasible)
        if distance is None:
            if result.success:
                raise ValueError("执行成功的路径不能使用不可达参考距离 None")
            continue
        reachable_count += 1
        if not result.success:
            continue
        length = result.executed_steps
        if type(length) is not int or length < distance:
            raise ValueError("成功路径长度必须为整数且不能短于参考最短距离")
        success_with_reference_count += 1
        optimal_success_count += length == distance
        optimality_ratio_sum += 1.0 if length == distance == 0 else distance / length

    return {
        "count": count,
        "success_count": success_count,
        "feasible_count": feasible_count,
        "reference_reachable_count": reachable_count,
        "reference_unreachable_count": count - reachable_count,
        "success_with_reference_count": success_with_reference_count,
        "optimal_success_count": optimal_success_count,
        "success_rate": _rate(success_count, count),
        "feasibility": _rate(feasible_count, count),
        "optimal_success_rate": _rate(optimal_success_count, reachable_count),
        "conditional_optimal_rate": _rate(optimal_success_count, success_with_reference_count),
        "mean_success_optimality_ratio": _rate(
            optimality_ratio_sum, success_with_reference_count
        ),
    }


def _truth_label(value: ExecutionResult | ParsedVerification | str) -> ParsedVerification:
    if isinstance(value, str):
        parsed = parse_verification_label(value)
        if parsed is None:
            raise ValueError(f"真实验证标签格式错误：{value!r}")
    elif isinstance(value, ParsedVerification):
        parsed = value
    else:
        parsed = ParsedVerification(
            valid=bool(value.success),
            step=value.first_error_step,
            reason=value.error_reason,
        )
    if parsed.valid:
        if parsed.step is not None or parsed.reason is not None:
            raise ValueError("有效的真实标签不能包含错误诊断")
    elif (
        type(parsed.step) is not int
        or parsed.step < 1
        or not isinstance(parsed.reason, ErrorReason)
    ):
        raise ValueError("无效的真实标签必须包含正整数步骤和已知错误类型")
    return parsed


def aggregate_verification_metrics(
    truths: Iterable[ExecutionResult | ParsedVerification | str],
    predictions: Iterable[str],
) -> dict:
    """评估有效性二分类与错误诊断，保留格式错误的预测。

    类别固定为 ``valid`` 和 ``invalid``。无法解析的预测视为未预测任何类别，
    为真实类别增加一次漏判，不为任何类别增加误报。某类 F1 分母为零时取零，
    仅整个数据集为空时 Macro-F1 才为 ``None``。全部诊断准确率均以所有真实无效
    候选为分母，包括被模型漏检的样本。
    """

    classes = {
        name: {"tp": 0, "fp": 0, "fn": 0, "support": 0}
        for name in ("valid", "invalid")
    }
    by_reason = {
        reason.value: {
            "count": 0,
            "first_error_localization_count": 0,
            "error_type_count": 0,
            "joint_diagnosis_count": 0,
        }
        for reason in ErrorReason
    }
    count = format_error_count = correct_count = invalid_count = 0
    localization_count = error_type_count = joint_count = 0
    for truth, prediction in zip(truths, predictions, strict=True):
        true = _truth_label(truth)
        predicted = parse_verification_label(prediction)
        count += 1
        format_error_count += predicted is None
        true_class = "valid" if true.valid else "invalid"
        predicted_class = (
            None if predicted is None else "valid" if predicted.valid else "invalid"
        )
        classes[true_class]["support"] += 1
        if predicted_class == true_class:
            classes[true_class]["tp"] += 1
            correct_count += 1
        else:
            classes[true_class]["fn"] += 1
            if predicted_class is not None:
                classes[predicted_class]["fp"] += 1
        if true.valid:
            continue
        invalid_count += 1
        diagnosed = predicted is not None and not predicted.valid
        step_correct = diagnosed and predicted.step == true.step
        reason_correct = diagnosed and predicted.reason == true.reason
        joint_correct = step_correct and reason_correct
        localization_count += step_correct
        error_type_count += reason_correct
        joint_count += joint_correct
        group = by_reason[true.reason.value]
        group["count"] += 1
        group["first_error_localization_count"] += step_correct
        group["error_type_count"] += reason_correct
        group["joint_diagnosis_count"] += joint_correct

    for statistics in classes.values():
        denominator = 2 * statistics["tp"] + statistics["fp"] + statistics["fn"]
        statistics["f1"] = 2 * statistics["tp"] / denominator if denominator else 0.0
    for statistics in by_reason.values():
        for prefix in ("first_error_localization", "error_type", "joint_diagnosis"):
            statistics[f"{prefix}_accuracy"] = _rate(
                statistics[f"{prefix}_count"], statistics["count"]
            )
    return {
        "count": count,
        "valid_count": count - invalid_count,
        "invalid_count": invalid_count,
        "correct_classification_count": correct_count,
        "classification_accuracy": _rate(correct_count, count),
        "format_error_count": format_error_count,
        "format_error_rate": _rate(format_error_count, count),
        "macro_f1": sum(statistics["f1"] for statistics in classes.values()) / 2
        if count
        else None,
        "first_error_localization_count": localization_count,
        "first_error_localization_accuracy": _rate(localization_count, invalid_count),
        "error_type_count": error_type_count,
        "error_type_accuracy": _rate(error_type_count, invalid_count),
        "joint_diagnosis_count": joint_count,
        "joint_diagnosis_accuracy": _rate(joint_count, invalid_count),
        "classes": classes,
        "by_reason": by_reason,
    }


def aggregate_repair_metrics(
    original_results: Iterable[ExecutionResult],
    final_results: Iterable[ExecutionResult],
) -> dict[str, int | float | None]:
    """按原始有效性确定修复率与破坏率分母，不依赖模型是否检出错误。"""

    count = original_invalid_count = repair_success_count = 0
    valid_destruction_count = final_success_count = 0
    for original, final in zip(original_results, final_results, strict=True):
        count += 1
        final_success_count += bool(final.success)
        if original.success:
            valid_destruction_count += not final.success
        else:
            original_invalid_count += 1
            repair_success_count += bool(final.success)
    original_valid_count = count - original_invalid_count
    return {
        "count": count,
        "original_invalid_count": original_invalid_count,
        "original_valid_count": original_valid_count,
        "repair_success_count": repair_success_count,
        "repair_success_rate": _rate(repair_success_count, original_invalid_count),
        "valid_destruction_count": valid_destruction_count,
        "valid_destruction_rate": _rate(valid_destruction_count, original_valid_count),
        "final_success_count": final_success_count,
        "final_success_rate": _rate(final_success_count, count),
    }
