import json

import numpy as np
import pandas as pd
import pytest

from ml_validation_toolkit import (
    ValidationResult,
    check_allowed_values,
    check_class_balance,
    check_duplicate_rows,
    check_infinite_values,
    check_missing_values,
    check_numeric_ranges,
    check_required_columns,
    run_data_checks,
    validation_passed,
)


@pytest.fixture
def valid_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_a": [0.1, 0.2, 0.3, 0.4],
            "feature_b": [10.0, 11.0, 12.0, 13.0],
            "label": ["normal", "normal", "defective", "defective"],
            "sample_id": ["a", "b", "c", "d"],
        }
    )


def test_validation_result_is_json_serializable() -> None:
    result = ValidationResult(
        name="example",
        passed=True,
        message="Example passed.",
        details={"count": 2},
    )

    serialized = json.dumps(result.to_dict())

    assert '"passed": true' in serialized


def test_required_columns_pass(
    valid_dataframe: pd.DataFrame,
) -> None:
    result = check_required_columns(
        valid_dataframe,
        ["feature_a", "label"],
    )

    assert result.passed is True
    assert result.details["missing_columns"] == []


def test_required_columns_fail(
    valid_dataframe: pd.DataFrame,
) -> None:
    result = check_required_columns(
        valid_dataframe,
        ["feature_a", "missing_column"],
    )

    assert result.passed is False
    assert result.details["missing_columns"] == ["missing_column"]


def test_missing_values_detected() -> None:
    dataframe = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, 4.0],
            "b": [1.0, 2.0, 3.0, 4.0],
        }
    )

    result = check_missing_values(
        dataframe,
        max_missing_fraction=0.0,
    )

    assert result.passed is False
    assert result.details["missing_counts"]["a"] == 1
    assert result.details["missing_fractions"]["a"] == 0.25


def test_missing_values_allowed_fraction() -> None:
    dataframe = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, 4.0],
        }
    )

    result = check_missing_values(
        dataframe,
        max_missing_fraction=0.25,
    )

    assert result.passed is True


def test_infinite_values_detected() -> None:
    dataframe = pd.DataFrame(
        {
            "a": [1.0, np.inf, 3.0],
            "b": [1.0, -np.inf, 3.0],
        }
    )

    result = check_infinite_values(dataframe)

    assert result.passed is False
    assert result.details["infinite_counts"]["a"] == 1
    assert result.details["infinite_counts"]["b"] == 1


def test_duplicate_rows_detected() -> None:
    dataframe = pd.DataFrame(
        {
            "sample_id": ["a", "b", "b"],
            "value": [1, 2, 2],
        }
    )

    result = check_duplicate_rows(
        dataframe,
        subset=["sample_id"],
    )

    assert result.passed is False
    assert result.details["duplicate_count"] == 1


def test_allowed_values_detect_unexpected_label() -> None:
    dataframe = pd.DataFrame(
        {
            "label": ["normal", "defective", "unknown"],
        }
    )

    result = check_allowed_values(
        dataframe,
        column="label",
        allowed_values=["normal", "defective"],
    )

    assert result.passed is False
    assert result.details["unexpected_values"] == ["unknown"]


def test_numeric_ranges_detect_violations() -> None:
    dataframe = pd.DataFrame(
        {
            "probability": [0.0, 0.4, 1.2],
            "temperature": [20.0, 25.0, 30.0],
        }
    )

    result = check_numeric_ranges(
        dataframe,
        ranges={
            "probability": (0.0, 1.0),
            "temperature": (0.0, 100.0),
        },
    )

    assert result.passed is False
    assert (
        result.details["violations"]["probability"]
        ["above_maximum_count"]
        == 1
    )
    assert "temperature" not in result.details["violations"]


def test_class_balance_passes(
    valid_dataframe: pd.DataFrame,
) -> None:
    result = check_class_balance(
        valid_dataframe,
        target_column="label",
        min_classes=2,
        min_samples_per_class=2,
        min_fraction_per_class=0.4,
    )

    assert result.passed is True
    assert result.details["class_counts"] == {
        "normal": 2,
        "defective": 2,
    }


def test_class_balance_detects_small_class() -> None:
    dataframe = pd.DataFrame(
        {
            "label": [
                "normal",
                "normal",
                "normal",
                "defective",
            ]
        }
    )

    result = check_class_balance(
        dataframe,
        target_column="label",
        min_samples_per_class=2,
    )

    assert result.passed is False
    assert result.details["low_count_classes"] == {
        "defective": 1,
    }


def test_run_data_checks_all_pass(
    valid_dataframe: pd.DataFrame,
) -> None:
    results = run_data_checks(
        valid_dataframe,
        required_columns=[
            "feature_a",
            "feature_b",
            "label",
            "sample_id",
        ],
        duplicate_subset=["sample_id"],
        allowed_values={
            "label": ["normal", "defective"],
        },
        numeric_ranges={
            "feature_a": (0.0, 1.0),
            "feature_b": (0.0, 20.0),
        },
        target_column="label",
        min_samples_per_class=2,
    )

    assert validation_passed(results) is True
    assert all(result.passed for result in results)


def test_run_data_checks_reports_failure(
    valid_dataframe: pd.DataFrame,
) -> None:
    invalid_dataframe = valid_dataframe.copy()
    invalid_dataframe.loc[0, "feature_a"] = np.inf

    results = run_data_checks(
        invalid_dataframe,
        required_columns=["feature_a", "label"],
        target_column="label",
    )

    assert validation_passed(results) is False

    failed_names = {
        result.name
        for result in results
        if not result.passed
    }

    assert "infinite_values" in failed_names


def test_invalid_missing_fraction_raises_error(
    valid_dataframe: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError):
        check_missing_values(
            valid_dataframe,
            max_missing_fraction=1.5,
        )
