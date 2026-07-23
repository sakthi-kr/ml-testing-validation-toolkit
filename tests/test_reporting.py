import json

import numpy as np
import pandas as pd
import pytest

from ml_validation_toolkit import (
    ValidationFailureError,
    ValidationResult,
    build_validation_report,
    format_validation_summary,
    print_validation_summary,
    raise_for_validation_failures,
    results_to_dataframe,
    results_to_records,
    save_validation_csv,
    save_validation_json,
    summarize_validation_results,
)


@pytest.fixture
def mixed_results() -> list[ValidationResult]:
    return [
        ValidationResult(
            name="required_columns",
            passed=True,
            message="All required columns are present.",
            details={
                "required_columns": ["score", "label"],
            },
        ),
        ValidationResult(
            name="score_range",
            passed=False,
            message="One score exceeds the maximum.",
            details={
                "maximum": 1.0,
                "observed_maximum": np.float64(1.4),
            },
        ),
    ]


def test_results_to_records(
    mixed_results: list[ValidationResult],
) -> None:
    records = results_to_records(mixed_results)

    assert len(records) == 2
    assert records[0]["status"] == "PASS"
    assert records[1]["status"] == "FAIL"
    assert records[1]["details"]["observed_maximum"] == 1.4


def test_summarize_validation_results(
    mixed_results: list[ValidationResult],
) -> None:
    summary = summarize_validation_results(
        mixed_results
    )

    assert summary["overall_passed"] is False
    assert summary["status"] == "FAIL"
    assert summary["total_checks"] == 2
    assert summary["passed_checks"] == 1
    assert summary["failed_checks"] == 1
    assert summary["failed_check_names"] == [
        "score_range"
    ]


def test_build_validation_report(
    mixed_results: list[ValidationResult],
) -> None:
    report = build_validation_report(
        mixed_results,
        report_name="Test Report",
        metadata={
            "model": "example-model",
            "version": np.int64(2),
        },
    )

    assert report["schema_version"] == "1.0"
    assert report["report_name"] == "Test Report"
    assert report["summary"]["status"] == "FAIL"
    assert report["metadata"]["version"] == 2
    assert len(report["checks"]) == 2
    assert report["generated_at_utc"].endswith("Z")


def test_results_to_dataframe(
    mixed_results: list[ValidationResult],
) -> None:
    dataframe = results_to_dataframe(
        mixed_results
    )

    assert isinstance(dataframe, pd.DataFrame)
    assert list(dataframe.columns) == [
        "check_index",
        "name",
        "passed",
        "status",
        "message",
        "details_json",
    ]

    assert dataframe.loc[0, "status"] == "PASS"
    assert dataframe.loc[1, "status"] == "FAIL"

    details = json.loads(
        dataframe.loc[1, "details_json"]
    )

    assert details["observed_maximum"] == 1.4


def test_format_validation_summary(
    mixed_results: list[ValidationResult],
) -> None:
    summary_text = format_validation_summary(
        mixed_results,
        report_name="Example Validation",
    )

    assert "Example Validation" in summary_text
    assert "Overall status : FAIL" in summary_text
    assert "[PASS] required_columns" in summary_text
    assert "[FAIL] score_range" in summary_text


def test_format_validation_summary_failures_only(
    mixed_results: list[ValidationResult],
) -> None:
    summary_text = format_validation_summary(
        mixed_results,
        report_name="Example Validation",
        include_passed_checks=False,
    )

    assert "[FAIL] score_range" in summary_text
    assert "[PASS] required_columns" not in summary_text


def test_print_validation_summary(
    mixed_results: list[ValidationResult],
    capsys: pytest.CaptureFixture[str],
) -> None:
    print_validation_summary(
        mixed_results,
        report_name="Console Report",
    )

    captured = capsys.readouterr()

    assert "Console Report" in captured.out
    assert "Overall status : FAIL" in captured.out


def test_save_validation_json(
    mixed_results: list[ValidationResult],
    tmp_path,
) -> None:
    output_path = (
        tmp_path
        / "reports"
        / "validation.json"
    )

    returned_path = save_validation_json(
        mixed_results,
        output_path,
        report_name="Saved JSON Report",
        metadata={
            "dataset": "test-data",
        },
    )

    assert returned_path == output_path
    assert output_path.exists()

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        report = json.load(file)

    assert report["report_name"] == "Saved JSON Report"
    assert report["metadata"]["dataset"] == "test-data"
    assert report["summary"]["failed_checks"] == 1


def test_save_validation_csv(
    mixed_results: list[ValidationResult],
    tmp_path,
) -> None:
    output_path = (
        tmp_path
        / "reports"
        / "validation.csv"
    )

    returned_path = save_validation_csv(
        mixed_results,
        output_path,
    )

    assert returned_path == output_path
    assert output_path.exists()

    dataframe = pd.read_csv(output_path)

    assert len(dataframe) == 2
    assert set(dataframe["status"]) == {
        "PASS",
        "FAIL",
    }


def test_raise_for_validation_failures(
    mixed_results: list[ValidationResult],
) -> None:
    with pytest.raises(
        ValidationFailureError,
        match="score_range",
    ):
        raise_for_validation_failures(
            mixed_results
        )


def test_raise_for_validation_failures_does_nothing() -> None:
    passing_results = [
        ValidationResult(
            name="example",
            passed=True,
            message="Passed.",
        )
    ]

    raise_for_validation_failures(
        passing_results
    )


def test_empty_results_generate_passing_report() -> None:
    report = build_validation_report(
        [],
        report_name="Empty Report",
    )

    assert report["summary"]["overall_passed"] is True
    assert report["summary"]["total_checks"] == 0
    assert report["checks"] == []


def test_invalid_result_type_raises_error() -> None:
    with pytest.raises(TypeError):
        results_to_records(
            [
                ValidationResult(
                    name="valid",
                    passed=True,
                    message="Passed.",
                ),
                "not-a-result",
            ]
        )


def test_non_finite_values_are_strict_json_compatible() -> None:
    results = [
        ValidationResult(
            name="non_finite",
            passed=False,
            message="Non-finite values found.",
            details={
                "nan": np.nan,
                "positive_infinity": np.inf,
                "negative_infinity": -np.inf,
            },
        )
    ]

    report = build_validation_report(results)

    serialized = json.dumps(
        report,
        allow_nan=False,
    )

    assert '"NaN"' in serialized
    assert '"Infinity"' in serialized
    assert '"-Infinity"' in serialized
