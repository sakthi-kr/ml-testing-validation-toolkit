"""
Reusable validation checks for machine-learning outputs and metrics.

The checks return structured ValidationResult objects so failures can be
reported, saved, or used to stop a machine-learning pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from ml_validation_toolkit.data_checks import ValidationResult


def _to_array(
    values: Sequence[Any] | np.ndarray | pd.Series,
    name: str,
) -> np.ndarray:
    """
    Convert an array-like input into a NumPy array.
    """
    if isinstance(values, (str, bytes)):
        raise TypeError(
            f"{name} must be an array-like collection, not a string."
        )

    if isinstance(values, pd.Series):
        array = values.to_numpy()
    else:
        array = np.asarray(values)

    if array.ndim == 0:
        raise TypeError(
            f"{name} must contain multiple values, not a scalar."
        )

    return array


def _to_1d_array(
    values: Sequence[Any] | np.ndarray | pd.Series,
    name: str,
) -> np.ndarray:
    """
    Convert an input into a one-dimensional NumPy array.
    """
    array = _to_array(values, name=name)

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional, "
            f"but received shape {array.shape}."
        )

    return array


def _python_value(value: Any) -> Any:
    """
    Convert NumPy scalar values into ordinary Python values.
    """
    if isinstance(value, np.generic):
        return value.item()

    return value


def _ordered_unique(values: Sequence[Any] | np.ndarray) -> list[Any]:
    """
    Return unique non-missing values while preserving order.
    """
    unique_values: list[Any] = []

    for value in values:
        value = _python_value(value)

        if pd.isna(value):
            continue

        if value not in unique_values:
            unique_values.append(value)

    return unique_values


def check_prediction_lengths(
    y_true: Sequence[Any] | np.ndarray | pd.Series,
    y_pred: Sequence[Any] | np.ndarray | pd.Series,
    additional_outputs: Mapping[
        str,
        Sequence[Any] | np.ndarray | pd.Series,
    ]
    | None = None,
) -> ValidationResult:
    """
    Check that predictions and optional model outputs have equal lengths.

    Parameters
    ----------
    y_true:
        Ground-truth labels.
    y_pred:
        Predicted labels.
    additional_outputs:
        Optional arrays that should contain one entry per sample, such as
        anomaly scores or confidence values.
    """
    true_array = _to_1d_array(y_true, "y_true")
    prediction_array = _to_1d_array(y_pred, "y_pred")

    lengths = {
        "y_true": int(len(true_array)),
        "y_pred": int(len(prediction_array)),
    }

    if additional_outputs is not None:
        for output_name, values in additional_outputs.items():
            output_array = _to_array(
                values,
                name=output_name,
            )

            lengths[str(output_name)] = int(len(output_array))

    expected_length = lengths["y_true"]

    mismatched_outputs = {
        name: length
        for name, length in lengths.items()
        if length != expected_length
    }

    passed = len(mismatched_outputs) == 0

    if passed:
        message = (
            f"All model outputs contain {expected_length} sample(s)."
        )
    else:
        message = (
            "Model-output lengths are inconsistent. "
            f"Expected {expected_length} sample(s); "
            f"mismatches: {mismatched_outputs}"
        )

    return ValidationResult(
        name="prediction_lengths",
        passed=passed,
        message=message,
        details={
            "expected_length": expected_length,
            "output_lengths": lengths,
            "mismatched_outputs": mismatched_outputs,
        },
    )


def check_prediction_labels(
    y_pred: Sequence[Any] | np.ndarray | pd.Series,
    allowed_labels: Sequence[Any],
    y_true: Sequence[Any] | np.ndarray | pd.Series | None = None,
    allow_missing: bool = False,
) -> ValidationResult:
    """
    Check that predicted and optional true labels are valid.

    Parameters
    ----------
    y_pred:
        Predicted labels.
    allowed_labels:
        Labels that may appear.
    y_true:
        Optional ground-truth labels to validate using the same label set.
    allow_missing:
        Whether missing label values are permitted.
    """
    prediction_array = _to_1d_array(y_pred, "y_pred")
    allowed = list(dict.fromkeys(allowed_labels))

    if not allowed:
        raise ValueError("allowed_labels cannot be empty.")

    arrays = {
        "y_pred": prediction_array,
    }

    if y_true is not None:
        arrays["y_true"] = _to_1d_array(y_true, "y_true")

    unexpected_values: dict[str, list[Any]] = {}
    missing_counts: dict[str, int] = {}
    observed_values: dict[str, list[Any]] = {}

    for array_name, array in arrays.items():
        missing_mask = pd.isna(array)
        missing_count = int(np.asarray(missing_mask).sum())

        observed = _ordered_unique(array)

        unexpected = [
            value
            for value in observed
            if value not in allowed
        ]

        observed_values[array_name] = observed
        missing_counts[array_name] = missing_count

        if unexpected:
            unexpected_values[array_name] = unexpected

    disallowed_missing = {
        name: count
        for name, count in missing_counts.items()
        if count > 0 and not allow_missing
    }

    passed = (
        len(unexpected_values) == 0
        and len(disallowed_missing) == 0
    )

    if passed:
        message = "All labels are within the configured allowed set."
    else:
        problems = []

        if unexpected_values:
            problems.append(
                f"unexpected labels: {unexpected_values}"
            )

        if disallowed_missing:
            problems.append(
                f"missing labels: {disallowed_missing}"
            )

        message = (
            "Label validation failed: "
            + "; ".join(problems)
        )

    return ValidationResult(
        name="prediction_labels",
        passed=passed,
        message=message,
        details={
            "allowed_labels": allowed,
            "observed_values": observed_values,
            "unexpected_values": unexpected_values,
            "missing_counts": missing_counts,
            "allow_missing": allow_missing,
        },
    )


def check_probability_matrix(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    expected_samples: int | None = None,
    expected_classes: int | None = None,
    row_sum_tolerance: float = 1e-6,
) -> ValidationResult:
    """
    Validate a classification-probability matrix.

    Checks:

    - matrix is two-dimensional
    - expected row and column counts
    - all values are finite
    - values are between zero and one
    - each row sums to approximately one
    """
    if row_sum_tolerance < 0:
        raise ValueError("row_sum_tolerance cannot be negative.")

    try:
        probability_array = np.asarray(
            probabilities,
            dtype=float,
        )
    except (TypeError, ValueError) as error:
        raise TypeError(
            "probabilities must contain numeric values."
        ) from error

    if probability_array.ndim != 2:
        return ValidationResult(
            name="probability_matrix",
            passed=False,
            message=(
                "Probability output must be a two-dimensional matrix. "
                f"Received shape {probability_array.shape}."
            ),
            details={
                "shape": list(probability_array.shape),
                "expected_samples": expected_samples,
                "expected_classes": expected_classes,
            },
        )

    sample_count, class_count = probability_array.shape

    shape_violations: dict[str, dict[str, int]] = {}

    if (
        expected_samples is not None
        and sample_count != expected_samples
    ):
        shape_violations["samples"] = {
            "expected": int(expected_samples),
            "observed": int(sample_count),
        }

    if (
        expected_classes is not None
        and class_count != expected_classes
    ):
        shape_violations["classes"] = {
            "expected": int(expected_classes),
            "observed": int(class_count),
        }

    finite_mask = np.isfinite(probability_array)
    non_finite_count = int((~finite_mask).sum())

    below_zero_count = int(
        (
            finite_mask
            & (probability_array < -row_sum_tolerance)
        ).sum()
    )

    above_one_count = int(
        (
            finite_mask
            & (probability_array > 1.0 + row_sum_tolerance)
        ).sum()
    )

    row_sums = probability_array.sum(axis=1)

    invalid_row_sum_mask = ~np.isclose(
        row_sums,
        1.0,
        atol=row_sum_tolerance,
        rtol=0.0,
    )

    invalid_row_sum_indices = [
        int(index)
        for index in np.where(invalid_row_sum_mask)[0]
    ]

    passed = (
        len(shape_violations) == 0
        and non_finite_count == 0
        and below_zero_count == 0
        and above_one_count == 0
        and len(invalid_row_sum_indices) == 0
    )

    if passed:
        message = (
            f"Probability matrix with shape "
            f"{probability_array.shape} is valid."
        )
    else:
        problems = []

        if shape_violations:
            problems.append(
                f"shape violations: {shape_violations}"
            )

        if non_finite_count > 0:
            problems.append(
                f"non-finite values: {non_finite_count}"
            )

        if below_zero_count > 0:
            problems.append(
                f"values below zero: {below_zero_count}"
            )

        if above_one_count > 0:
            problems.append(
                f"values above one: {above_one_count}"
            )

        if invalid_row_sum_indices:
            problems.append(
                "rows not summing to one: "
                f"{invalid_row_sum_indices}"
            )

        message = (
            "Probability-matrix validation failed: "
            + "; ".join(problems)
        )

    finite_values = probability_array[finite_mask]

    return ValidationResult(
        name="probability_matrix",
        passed=passed,
        message=message,
        details={
            "shape": [int(sample_count), int(class_count)],
            "expected_samples": expected_samples,
            "expected_classes": expected_classes,
            "shape_violations": shape_violations,
            "non_finite_count": non_finite_count,
            "below_zero_count": below_zero_count,
            "above_one_count": above_one_count,
            "row_sum_tolerance": row_sum_tolerance,
            "invalid_row_sum_indices": invalid_row_sum_indices,
            "minimum_probability": (
                float(finite_values.min())
                if finite_values.size > 0
                else None
            ),
            "maximum_probability": (
                float(finite_values.max())
                if finite_values.size > 0
                else None
            ),
        },
    )


def check_score_range(
    scores: Sequence[float] | np.ndarray | pd.Series,
    minimum: float | None = 0.0,
    maximum: float | None = 1.0,
    allow_missing: bool = False,
    inclusive: bool = True,
) -> ValidationResult:
    """
    Validate one-dimensional confidence, probability, or anomaly scores.

    Bounds can be set to None when the score is unbounded on that side.
    """
    if (
        minimum is not None
        and maximum is not None
        and minimum > maximum
    ):
        raise ValueError(
            "minimum cannot be greater than maximum."
        )

    try:
        score_array = _to_1d_array(scores, "scores").astype(float)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "scores must contain numeric values."
        ) from error

    missing_mask = np.isnan(score_array)
    infinite_mask = np.isinf(score_array)

    valid_mask = ~(missing_mask | infinite_mask)
    valid_scores = score_array[valid_mask]

    if minimum is None:
        below_minimum_mask = np.zeros(
            valid_scores.shape,
            dtype=bool,
        )
    elif inclusive:
        below_minimum_mask = valid_scores < minimum
    else:
        below_minimum_mask = valid_scores <= minimum

    if maximum is None:
        above_maximum_mask = np.zeros(
            valid_scores.shape,
            dtype=bool,
        )
    elif inclusive:
        above_maximum_mask = valid_scores > maximum
    else:
        above_maximum_mask = valid_scores >= maximum

    missing_count = int(missing_mask.sum())
    infinite_count = int(infinite_mask.sum())
    below_minimum_count = int(below_minimum_mask.sum())
    above_maximum_count = int(above_maximum_mask.sum())

    passed = (
        (allow_missing or missing_count == 0)
        and infinite_count == 0
        and below_minimum_count == 0
        and above_maximum_count == 0
    )

    if passed:
        message = "All scores are within the configured range."
    else:
        problems = []

        if missing_count > 0 and not allow_missing:
            problems.append(
                f"missing scores: {missing_count}"
            )

        if infinite_count > 0:
            problems.append(
                f"infinite scores: {infinite_count}"
            )

        if below_minimum_count > 0:
            problems.append(
                f"scores below minimum: {below_minimum_count}"
            )

        if above_maximum_count > 0:
            problems.append(
                f"scores above maximum: {above_maximum_count}"
            )

        message = (
            "Score-range validation failed: "
            + "; ".join(problems)
        )

    return ValidationResult(
        name="score_range",
        passed=passed,
        message=message,
        details={
            "minimum": minimum,
            "maximum": maximum,
            "inclusive": inclusive,
            "allow_missing": allow_missing,
            "missing_count": missing_count,
            "infinite_count": infinite_count,
            "below_minimum_count": below_minimum_count,
            "above_maximum_count": above_maximum_count,
            "observed_minimum": (
                float(valid_scores.min())
                if valid_scores.size > 0
                else None
            ),
            "observed_maximum": (
                float(valid_scores.max())
                if valid_scores.size > 0
                else None
            ),
        },
    )


def check_metric_thresholds(
    metrics: Mapping[str, float],
    minimums: Mapping[str, float] | None = None,
    maximums: Mapping[str, float] | None = None,
) -> ValidationResult:
    """
    Check model metrics against configured minimum and maximum thresholds.

    Example
    -------
    minimums={
        "accuracy": 0.90,
        "f1_score": 0.85,
    }
    """
    minimums = dict(minimums or {})
    maximums = dict(maximums or {})

    if not minimums and not maximums:
        raise ValueError(
            "At least one minimum or maximum threshold is required."
        )

    metric_values = dict(metrics)

    required_metric_names = list(
        dict.fromkeys(
            [*minimums.keys(), *maximums.keys()]
        )
    )

    missing_metrics = [
        metric_name
        for metric_name in required_metric_names
        if metric_name not in metric_values
    ]

    non_numeric_metrics: dict[str, Any] = {}
    non_finite_metrics: dict[str, Any] = {}
    violations: dict[str, dict[str, float]] = {}

    for metric_name in required_metric_names:
        if metric_name not in metric_values:
            continue

        value = metric_values[metric_name]

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            non_numeric_metrics[metric_name] = value
            continue

        if not np.isfinite(numeric_value):
            non_finite_metrics[metric_name] = numeric_value
            continue

        metric_violation: dict[str, float] = {
            "observed": numeric_value,
        }

        failed = False

        if metric_name in minimums:
            minimum_value = float(minimums[metric_name])
            metric_violation["minimum"] = minimum_value

            if numeric_value < minimum_value:
                failed = True

        if metric_name in maximums:
            maximum_value = float(maximums[metric_name])
            metric_violation["maximum"] = maximum_value

            if numeric_value > maximum_value:
                failed = True

        if failed:
            violations[metric_name] = metric_violation

    passed = (
        len(missing_metrics) == 0
        and len(non_numeric_metrics) == 0
        and len(non_finite_metrics) == 0
        and len(violations) == 0
    )

    if passed:
        message = (
            f"All {len(required_metric_names)} configured "
            "metric threshold checks passed."
        )
    else:
        problems = []

        if missing_metrics:
            problems.append(
                f"missing metrics: {missing_metrics}"
            )

        if non_numeric_metrics:
            problems.append(
                f"non-numeric metrics: {non_numeric_metrics}"
            )

        if non_finite_metrics:
            problems.append(
                f"non-finite metrics: {non_finite_metrics}"
            )

        if violations:
            problems.append(
                f"threshold violations: {violations}"
            )

        message = (
            "Metric-threshold validation failed: "
            + "; ".join(problems)
        )

    return ValidationResult(
        name="metric_thresholds",
        passed=passed,
        message=message,
        details={
            "metrics": {
                name: _python_value(value)
                for name, value in metric_values.items()
            },
            "minimums": minimums,
            "maximums": maximums,
            "missing_metrics": missing_metrics,
            "non_numeric_metrics": non_numeric_metrics,
            "non_finite_metrics": non_finite_metrics,
            "violations": violations,
        },
    )


def check_confusion_matrix_consistency(
    y_true: Sequence[Any] | np.ndarray | pd.Series,
    y_pred: Sequence[Any] | np.ndarray | pd.Series,
    matrix: Sequence[Sequence[int]] | np.ndarray | pd.DataFrame,
    labels: Sequence[Any] | None = None,
) -> ValidationResult:
    """
    Check that a supplied confusion matrix matches the predictions.
    """
    true_array = _to_1d_array(y_true, "y_true")
    prediction_array = _to_1d_array(y_pred, "y_pred")

    if len(true_array) != len(prediction_array):
        return ValidationResult(
            name="confusion_matrix_consistency",
            passed=False,
            message=(
                "Cannot validate confusion matrix because y_true and "
                "y_pred have different lengths."
            ),
            details={
                "y_true_length": int(len(true_array)),
                "y_pred_length": int(len(prediction_array)),
            },
        )

    if labels is None:
        combined_values = np.concatenate(
            [true_array, prediction_array]
        )
        matrix_labels = _ordered_unique(combined_values)
    else:
        matrix_labels = list(dict.fromkeys(labels))

    if not matrix_labels:
        raise ValueError("At least one confusion-matrix label is required.")

    try:
        observed_matrix = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "matrix must contain numeric values."
        ) from error

    expected_shape = (
        len(matrix_labels),
        len(matrix_labels),
    )

    if observed_matrix.ndim != 2:
        return ValidationResult(
            name="confusion_matrix_consistency",
            passed=False,
            message=(
                "Confusion matrix must be two-dimensional. "
                f"Received shape {observed_matrix.shape}."
            ),
            details={
                "labels": matrix_labels,
                "expected_shape": list(expected_shape),
                "observed_shape": list(observed_matrix.shape),
            },
        )

    if observed_matrix.shape != expected_shape:
        return ValidationResult(
            name="confusion_matrix_consistency",
            passed=False,
            message=(
                "Confusion-matrix shape does not match the label set. "
                f"Expected {expected_shape}, "
                f"received {observed_matrix.shape}."
            ),
            details={
                "labels": matrix_labels,
                "expected_shape": list(expected_shape),
                "observed_shape": list(observed_matrix.shape),
            },
        )

    finite = bool(np.isfinite(observed_matrix).all())
    non_negative = bool((observed_matrix >= 0).all())
    integer_valued = bool(
        np.allclose(
            observed_matrix,
            np.round(observed_matrix),
        )
    )

    expected_matrix = confusion_matrix(
        true_array,
        prediction_array,
        labels=matrix_labels,
    )

    matrices_equal = bool(
        np.array_equal(
            np.round(observed_matrix).astype(int),
            expected_matrix,
        )
    )

    observed_total = float(observed_matrix.sum())
    expected_total = int(len(true_array))

    total_matches = bool(
        np.isclose(
            observed_total,
            expected_total,
        )
    )

    passed = (
        finite
        and non_negative
        and integer_valued
        and matrices_equal
        and total_matches
    )

    if passed:
        message = (
            "The supplied confusion matrix matches the predictions."
        )
    else:
        problems = []

        if not finite:
            problems.append("matrix contains non-finite values")

        if not non_negative:
            problems.append("matrix contains negative values")

        if not integer_valued:
            problems.append("matrix contains non-integer values")

        if not matrices_equal:
            problems.append(
                "matrix values do not match y_true and y_pred"
            )

        if not total_matches:
            problems.append(
                "matrix total does not match sample count"
            )

        message = (
            "Confusion-matrix validation failed: "
            + "; ".join(problems)
        )

    return ValidationResult(
        name="confusion_matrix_consistency",
        passed=passed,
        message=message,
        details={
            "labels": matrix_labels,
            "expected_matrix": expected_matrix.tolist(),
            "observed_matrix": observed_matrix.tolist(),
            "expected_total": expected_total,
            "observed_total": observed_total,
            "finite": finite,
            "non_negative": non_negative,
            "integer_valued": integer_valued,
            "matrices_equal": matrices_equal,
            "total_matches": total_matches,
        },
    )


def validation_passed(
    results: Sequence[ValidationResult],
) -> bool:
    """
    Return True only when every model-validation result passed.
    """
    return all(result.passed for result in results)


def run_model_checks(
    *,
    y_true: Sequence[Any] | np.ndarray | pd.Series,
    y_pred: Sequence[Any] | np.ndarray | pd.Series,
    allowed_labels: Sequence[Any] | None = None,
    probabilities: (
        Sequence[Sequence[float]] | np.ndarray | None
    ) = None,
    expected_probability_classes: int | None = None,
    scores: Sequence[float] | np.ndarray | pd.Series | None = None,
    score_minimum: float | None = 0.0,
    score_maximum: float | None = 1.0,
    metrics: Mapping[str, float] | None = None,
    metric_minimums: Mapping[str, float] | None = None,
    metric_maximums: Mapping[str, float] | None = None,
    matrix: (
        Sequence[Sequence[int]]
        | np.ndarray
        | pd.DataFrame
        | None
    ) = None,
    confusion_matrix_labels: Sequence[Any] | None = None,
) -> list[ValidationResult]:
    """
    Run a standard collection of model-output validation checks.
    """
    true_array = _to_1d_array(y_true, "y_true")
    prediction_array = _to_1d_array(y_pred, "y_pred")

    additional_outputs: dict[str, Any] = {}

    if probabilities is not None:
        additional_outputs["probabilities"] = probabilities

    if scores is not None:
        additional_outputs["scores"] = scores

    results = [
        check_prediction_lengths(
            y_true=true_array,
            y_pred=prediction_array,
            additional_outputs=additional_outputs,
        )
    ]

    if allowed_labels is not None:
        results.append(
            check_prediction_labels(
                y_pred=prediction_array,
                y_true=true_array,
                allowed_labels=allowed_labels,
            )
        )

    if probabilities is not None:
        results.append(
            check_probability_matrix(
                probabilities=probabilities,
                expected_samples=len(true_array),
                expected_classes=expected_probability_classes,
            )
        )

    if scores is not None:
        results.append(
            check_score_range(
                scores=scores,
                minimum=score_minimum,
                maximum=score_maximum,
            )
        )

    if (
        metrics is not None
        and (
            metric_minimums is not None
            or metric_maximums is not None
        )
    ):
        results.append(
            check_metric_thresholds(
                metrics=metrics,
                minimums=metric_minimums,
                maximums=metric_maximums,
            )
        )

    if matrix is not None:
        results.append(
            check_confusion_matrix_consistency(
                y_true=true_array,
                y_pred=prediction_array,
                matrix=matrix,
                labels=confusion_matrix_labels,
            )
        )

    return results
