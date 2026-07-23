"""
Reporting utilities for machine-learning validation results.

This module converts ValidationResult objects into:

- human-readable console summaries
- JSON validation reports
- CSV validation tables
- pipeline exceptions when validation fails
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml_validation_toolkit.data_checks import ValidationResult


class ValidationFailureError(RuntimeError):
    """
    Raised when one or more configured validation checks fail.
    """


def _validate_results(
    results: Sequence[ValidationResult],
) -> list[ValidationResult]:
    """
    Validate and normalize a collection of ValidationResult objects.
    """
    if isinstance(results, (str, bytes)):
        raise TypeError(
            "results must be a sequence of ValidationResult objects."
        )

    normalized_results = list(results)

    invalid_types = [
        type(result).__name__
        for result in normalized_results
        if not isinstance(result, ValidationResult)
    ]

    if invalid_types:
        raise TypeError(
            "Every result must be a ValidationResult. "
            f"Invalid types: {invalid_types}"
        )

    return normalized_results


def _json_compatible(value: Any) -> Any:
    """
    Convert common scientific-Python objects into strict JSON values.
    """
    if value is None:
        return None

    if value is pd.NA:
        return None

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (datetime, date, time)):
        return value.isoformat()

    if isinstance(value, np.ndarray):
        return [
            _json_compatible(item)
            for item in value.tolist()
        ]

    if isinstance(value, np.generic):
        return _json_compatible(value.item())

    if isinstance(value, pd.Series):
        return [
            _json_compatible(item)
            for item in value.tolist()
        ]

    if isinstance(value, pd.Index):
        return [
            _json_compatible(item)
            for item in value.tolist()
        ]

    if isinstance(value, Mapping):
        return {
            str(key): _json_compatible(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _json_compatible(item)
            for item in value
        ]

    if isinstance(value, set):
        return [
            _json_compatible(item)
            for item in sorted(
                value,
                key=lambda item: str(item),
            )
        ]

    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"

        if math.isinf(value):
            return (
                "Infinity"
                if value > 0
                else "-Infinity"
            )

        return value

    if isinstance(value, (str, int, bool)):
        return value

    return str(value)


def results_to_records(
    results: Sequence[ValidationResult],
) -> list[dict[str, Any]]:
    """
    Convert validation results into JSON-compatible dictionaries.
    """
    normalized_results = _validate_results(results)

    records: list[dict[str, Any]] = []

    for index, result in enumerate(normalized_results):
        records.append(
            {
                "check_index": index,
                "name": result.name,
                "passed": bool(result.passed),
                "status": (
                    "PASS"
                    if result.passed
                    else "FAIL"
                ),
                "message": result.message,
                "details": _json_compatible(result.details),
            }
        )

    return records


def summarize_validation_results(
    results: Sequence[ValidationResult],
) -> dict[str, Any]:
    """
    Calculate overall validation counts and failed-check names.
    """
    normalized_results = _validate_results(results)

    passed_results = [
        result
        for result in normalized_results
        if result.passed
    ]

    failed_results = [
        result
        for result in normalized_results
        if not result.passed
    ]

    return {
        "overall_passed": len(failed_results) == 0,
        "status": (
            "PASS"
            if len(failed_results) == 0
            else "FAIL"
        ),
        "total_checks": len(normalized_results),
        "passed_checks": len(passed_results),
        "failed_checks": len(failed_results),
        "passed_check_names": [
            result.name
            for result in passed_results
        ],
        "failed_check_names": [
            result.name
            for result in failed_results
        ],
    }


def build_validation_report(
    results: Sequence[ValidationResult],
    *,
    report_name: str = "ML Validation Report",
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a complete structured validation report.
    """
    if not report_name.strip():
        raise ValueError("report_name cannot be empty.")

    normalized_results = _validate_results(results)

    generated_at = datetime.now(
        timezone.utc
    ).isoformat().replace("+00:00", "Z")

    return {
        "schema_version": "1.0",
        "report_name": report_name,
        "generated_at_utc": generated_at,
        "summary": summarize_validation_results(
            normalized_results
        ),
        "metadata": _json_compatible(
            dict(metadata or {})
        ),
        "checks": results_to_records(
            normalized_results
        ),
    }


def results_to_dataframe(
    results: Sequence[ValidationResult],
) -> pd.DataFrame:
    """
    Convert validation results into a flat dataframe.

    Structured details are stored as compact JSON strings so the table can be
    exported to CSV without losing nested validation information.
    """
    records = results_to_records(results)

    rows = []

    for record in records:
        rows.append(
            {
                "check_index": record["check_index"],
                "name": record["name"],
                "passed": record["passed"],
                "status": record["status"],
                "message": record["message"],
                "details_json": json.dumps(
                    record["details"],
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "check_index",
            "name",
            "passed",
            "status",
            "message",
            "details_json",
        ],
    )


def format_validation_summary(
    results: Sequence[ValidationResult],
    *,
    report_name: str = "ML Validation Report",
    include_passed_checks: bool = True,
) -> str:
    """
    Create a human-readable multiline validation summary.
    """
    if not report_name.strip():
        raise ValueError("report_name cannot be empty.")

    normalized_results = _validate_results(results)
    summary = summarize_validation_results(
        normalized_results
    )

    lines = [
        report_name,
        "=" * len(report_name),
        f"Overall status : {summary['status']}",
        f"Total checks   : {summary['total_checks']}",
        f"Passed checks  : {summary['passed_checks']}",
        f"Failed checks  : {summary['failed_checks']}",
    ]

    visible_results = [
        result
        for result in normalized_results
        if include_passed_checks or not result.passed
    ]

    if visible_results:
        lines.append("")
        lines.append("Check results")
        lines.append("-------------")

        for result in visible_results:
            status = (
                "PASS"
                if result.passed
                else "FAIL"
            )

            lines.append(
                f"[{status}] {result.name}: "
                f"{result.message}"
            )

    return "\n".join(lines)


def print_validation_summary(
    results: Sequence[ValidationResult],
    *,
    report_name: str = "ML Validation Report",
    include_passed_checks: bool = True,
) -> None:
    """
    Print a human-readable validation summary.
    """
    print(
        format_validation_summary(
            results,
            report_name=report_name,
            include_passed_checks=include_passed_checks,
        )
    )


def save_validation_json(
    results: Sequence[ValidationResult],
    output_path: str | Path,
    *,
    report_name: str = "ML Validation Report",
    metadata: Mapping[str, Any] | None = None,
    indent: int = 4,
) -> Path:
    """
    Save a complete validation report as JSON.
    """
    if indent < 0:
        raise ValueError("indent cannot be negative.")

    destination = Path(output_path)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = build_validation_report(
        results,
        report_name=report_name,
        metadata=metadata,
    )

    with destination.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=indent,
            ensure_ascii=False,
            allow_nan=False,
        )

    return destination


def save_validation_csv(
    results: Sequence[ValidationResult],
    output_path: str | Path,
) -> Path:
    """
    Save validation checks as a flat CSV table.
    """
    destination = Path(output_path)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = results_to_dataframe(results)

    dataframe.to_csv(
        destination,
        index=False,
    )

    return destination


def raise_for_validation_failures(
    results: Sequence[ValidationResult],
    *,
    message_prefix: str = "ML validation failed",
) -> None:
    """
    Raise ValidationFailureError when any check has failed.

    This function is intended for automated pipelines and continuous
    integration workflows where failed validation should stop execution.
    """
    normalized_results = _validate_results(results)

    failed_results = [
        result
        for result in normalized_results
        if not result.passed
    ]

    if not failed_results:
        return

    failure_descriptions = [
        f"{result.name}: {result.message}"
        for result in failed_results
    ]

    formatted_failures = "\n".join(
        f"- {description}"
        for description in failure_descriptions
    )

    raise ValidationFailureError(
        f"{message_prefix}.\n"
        f"{formatted_failures}"
    )
