import numpy as np
import pandas as pd
import pytest

from ml_validation_toolkit import (
    check_confusion_matrix_consistency,
    check_metric_thresholds,
    check_prediction_labels,
    check_prediction_lengths,
    check_probability_matrix,
    check_score_range,
    run_model_checks,
    validation_passed,
)


@pytest.fixture
def binary_predictions() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])

    probabilities = np.array(
        [
            [0.90, 0.10],
            [0.40, 0.60],
            [0.20, 0.80],
            [0.10, 0.90],
        ]
    )

    return y_true, y_pred, probabilities


def test_prediction_lengths_pass(
    binary_predictions: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
) -> None:
    y_true, y_pred, probabilities = binary_predictions

    result = check_prediction_lengths(
        y_true=y_true,
        y_pred=y_pred,
        additional_outputs={
            "probabilities": probabilities,
        },
    )

    assert result.passed is True
    assert result.details["expected_length"] == 4


def test_prediction_lengths_detect_mismatch() -> None:
    result = check_prediction_lengths(
        y_true=[0, 1, 1],
        y_pred=[0, 1],
    )

    assert result.passed is False
    assert result.details["mismatched_outputs"] == {
        "y_pred": 2,
    }


def test_prediction_labels_pass() -> None:
    result = check_prediction_labels(
        y_true=["normal", "defective"],
        y_pred=["normal", "normal"],
        allowed_labels=["normal", "defective"],
    )

    assert result.passed is True


def test_prediction_labels_detect_unknown_label() -> None:
    result = check_prediction_labels(
        y_pred=["normal", "unknown"],
        allowed_labels=["normal", "defective"],
    )

    assert result.passed is False
    assert result.details["unexpected_values"] == {
        "y_pred": ["unknown"],
    }


def test_probability_matrix_pass(
    binary_predictions: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
) -> None:
    _, _, probabilities = binary_predictions

    result = check_probability_matrix(
        probabilities,
        expected_samples=4,
        expected_classes=2,
    )

    assert result.passed is True
    assert result.details["shape"] == [4, 2]


def test_probability_matrix_detects_invalid_row_sum() -> None:
    probabilities = np.array(
        [
            [0.8, 0.2],
            [0.4, 0.8],
        ]
    )

    result = check_probability_matrix(probabilities)

    assert result.passed is False
    assert result.details["invalid_row_sum_indices"] == [1]


def test_probability_matrix_detects_out_of_range_value() -> None:
    probabilities = np.array(
        [
            [1.2, -0.2],
            [0.4, 0.6],
        ]
    )

    result = check_probability_matrix(probabilities)

    assert result.passed is False
    assert result.details["below_zero_count"] == 1
    assert result.details["above_one_count"] == 1


def test_score_range_passes() -> None:
    result = check_score_range(
        scores=[0.1, 0.5, 0.9],
        minimum=0.0,
        maximum=1.0,
    )

    assert result.passed is True


def test_score_range_detects_invalid_scores() -> None:
    result = check_score_range(
        scores=[0.1, -0.2, 1.4],
        minimum=0.0,
        maximum=1.0,
    )

    assert result.passed is False
    assert result.details["below_minimum_count"] == 1
    assert result.details["above_maximum_count"] == 1


def test_score_range_supports_unbounded_maximum() -> None:
    result = check_score_range(
        scores=[0.1, 2.5, 10.0],
        minimum=0.0,
        maximum=None,
    )

    assert result.passed is True


def test_metric_thresholds_pass() -> None:
    result = check_metric_thresholds(
        metrics={
            "accuracy": 0.95,
            "f1_score": 0.91,
            "latency_ms": 18.0,
        },
        minimums={
            "accuracy": 0.90,
            "f1_score": 0.85,
        },
        maximums={
            "latency_ms": 25.0,
        },
    )

    assert result.passed is True


def test_metric_thresholds_detect_failure() -> None:
    result = check_metric_thresholds(
        metrics={
            "accuracy": 0.82,
            "latency_ms": 31.0,
        },
        minimums={
            "accuracy": 0.90,
        },
        maximums={
            "latency_ms": 25.0,
        },
    )

    assert result.passed is False
    assert "accuracy" in result.details["violations"]
    assert "latency_ms" in result.details["violations"]


def test_metric_thresholds_detect_missing_metric() -> None:
    result = check_metric_thresholds(
        metrics={
            "accuracy": 0.95,
        },
        minimums={
            "accuracy": 0.90,
            "f1_score": 0.85,
        },
    )

    assert result.passed is False
    assert result.details["missing_metrics"] == ["f1_score"]


def test_confusion_matrix_consistency_passes(
    binary_predictions: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
) -> None:
    y_true, y_pred, _ = binary_predictions

    matrix = np.array(
        [
            [1, 1],
            [0, 2],
        ]
    )

    result = check_confusion_matrix_consistency(
        y_true=y_true,
        y_pred=y_pred,
        matrix=matrix,
        labels=[0, 1],
    )

    assert result.passed is True


def test_confusion_matrix_consistency_detects_error(
    binary_predictions: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
) -> None:
    y_true, y_pred, _ = binary_predictions

    incorrect_matrix = np.array(
        [
            [2, 0],
            [0, 2],
        ]
    )

    result = check_confusion_matrix_consistency(
        y_true=y_true,
        y_pred=y_pred,
        matrix=incorrect_matrix,
        labels=[0, 1],
    )

    assert result.passed is False
    assert result.details["matrices_equal"] is False


def test_confusion_matrix_accepts_dataframe(
    binary_predictions: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
) -> None:
    y_true, y_pred, _ = binary_predictions

    matrix = pd.DataFrame(
        [
            [1, 1],
            [0, 2],
        ]
    )

    result = check_confusion_matrix_consistency(
        y_true=y_true,
        y_pred=y_pred,
        matrix=matrix,
        labels=[0, 1],
    )

    assert result.passed is True


def test_run_model_checks_all_pass(
    binary_predictions: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
) -> None:
    y_true, y_pred, probabilities = binary_predictions

    scores = probabilities[:, 1]

    results = run_model_checks(
        y_true=y_true,
        y_pred=y_pred,
        allowed_labels=[0, 1],
        probabilities=probabilities,
        expected_probability_classes=2,
        scores=scores,
        score_minimum=0.0,
        score_maximum=1.0,
        metrics={
            "accuracy": 0.75,
            "f1_score": 0.80,
        },
        metric_minimums={
            "accuracy": 0.70,
            "f1_score": 0.75,
        },
        matrix=[
            [1, 1],
            [0, 2],
        ],
        confusion_matrix_labels=[0, 1],
    )

    assert validation_passed(results) is True
    assert all(result.passed for result in results)


def test_run_model_checks_reports_failure() -> None:
    results = run_model_checks(
        y_true=[0, 1, 1],
        y_pred=[0, 2, 1],
        allowed_labels=[0, 1],
        scores=[0.1, 0.8, 1.5],
        score_minimum=0.0,
        score_maximum=1.0,
    )

    assert validation_passed(results) is False

    failed_names = {
        result.name
        for result in results
        if not result.passed
    }

    assert "prediction_labels" in failed_names
    assert "score_range" in failed_names


def test_metric_thresholds_require_configuration() -> None:
    with pytest.raises(ValueError):
        check_metric_thresholds(
            metrics={"accuracy": 0.9},
        )
