"""
Reusable validation checks for pandas DataFrames.

The functions in this module return structured ValidationResult objects rather
than raising errors for ordinary validation failures. This allows projects to
save reports, print readable summaries, and decide whether a failed check
should stop a pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidationResult:
    """
    Result produced by one validation check.

    Attributes
    ----------
    name:
        Machine-readable check name.
    passed:
        True when the check passed.
    message:
        Human-readable explanation.
    details:
        Additional structured information for reporting.
    """

    name: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the result into a JSON-compatible dictionary.
        """
        return asdict(self)


def _ensure_dataframe(dataframe: pd.DataFrame) -> None:
    """
    Raise a clear error when the input is not a pandas DataFrame.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "Expected a pandas DataFrame, "
            f"but received {type(dataframe).__name__}."
        )


def _unique_list(values: Iterable[str]) -> list[str]:
    """
    Return unique strings while preserving their original order.
    """
    return list(dict.fromkeys(values))


def _find_missing_columns(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
) -> list[str]:
    """
    Find requested columns that are absent from the dataframe.
    """
    return [
        column
        for column in columns
        if column not in dataframe.columns
    ]


def check_required_columns(
    dataframe: pd.DataFrame,
    required_columns: Sequence[str],
) -> ValidationResult:
    """
    Check that all required columns are present.
    """
    _ensure_dataframe(dataframe)

    required = _unique_list(required_columns)
    missing_columns = _find_missing_columns(dataframe, required)

    passed = len(missing_columns) == 0

    if passed:
        message = f"All {len(required)} required columns are present."
    else:
        message = (
            f"{len(missing_columns)} required column(s) are missing: "
            f"{missing_columns}"
        )

    return ValidationResult(
        name="required_columns",
        passed=passed,
        message=message,
        details={
            "required_columns": required,
            "missing_columns": missing_columns,
        },
    )


def check_missing_values(
    dataframe: pd.DataFrame,
    columns: Sequence[str] | None = None,
    max_missing_fraction: float = 0.0,
) -> ValidationResult:
    """
    Check missing-value fractions for selected columns.

    Parameters
    ----------
    dataframe:
        Dataframe to validate.
    columns:
        Columns to inspect. When omitted, all columns are inspected.
    max_missing_fraction:
        Maximum allowed missing fraction in each inspected column.
        Must be between 0 and 1.
    """
    _ensure_dataframe(dataframe)

    if not 0.0 <= max_missing_fraction <= 1.0:
        raise ValueError(
            "max_missing_fraction must be between 0 and 1."
        )

    selected_columns = (
        list(dataframe.columns)
        if columns is None
        else _unique_list(columns)
    )

    missing_columns = _find_missing_columns(
        dataframe,
        selected_columns,
    )

    if missing_columns:
        return ValidationResult(
            name="missing_values",
            passed=False,
            message=(
                "Missing-value check could not inspect absent columns: "
                f"{missing_columns}"
            ),
            details={
                "inspected_columns": selected_columns,
                "missing_columns": missing_columns,
            },
        )

    if dataframe.empty:
        missing_counts = {
            column: 0
            for column in selected_columns
        }
        missing_fractions = {
            column: 0.0
            for column in selected_columns
        }
    else:
        missing_counts = {
            column: int(value)
            for column, value in (
                dataframe[selected_columns]
                .isna()
                .sum()
                .to_dict()
                .items()
            )
        }

        missing_fractions = {
            column: float(count / len(dataframe))
            for column, count in missing_counts.items()
        }

    violating_columns = {
        column: fraction
        for column, fraction in missing_fractions.items()
        if fraction > max_missing_fraction
    }

    passed = len(violating_columns) == 0

    if passed:
        message = (
            "Missing-value fractions are within the allowed limit "
            f"of {max_missing_fraction:.3f}."
        )
    else:
        message = (
            f"{len(violating_columns)} column(s) exceed the allowed "
            f"missing fraction of {max_missing_fraction:.3f}."
        )

    return ValidationResult(
        name="missing_values",
        passed=passed,
        message=message,
        details={
            "row_count": int(len(dataframe)),
            "inspected_columns": selected_columns,
            "missing_counts": missing_counts,
            "missing_fractions": missing_fractions,
            "violating_columns": violating_columns,
            "max_missing_fraction": max_missing_fraction,
        },
    )


def check_infinite_values(
    dataframe: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> ValidationResult:
    """
    Check selected numeric columns for positive or negative infinity.

    When columns are not supplied, every numeric dataframe column is checked.
    """
    _ensure_dataframe(dataframe)

    if columns is None:
        selected_columns = list(
            dataframe.select_dtypes(include=[np.number]).columns
        )
    else:
        selected_columns = _unique_list(columns)

    missing_columns = _find_missing_columns(
        dataframe,
        selected_columns,
    )

    if missing_columns:
        return ValidationResult(
            name="infinite_values",
            passed=False,
            message=(
                "Infinite-value check could not inspect absent columns: "
                f"{missing_columns}"
            ),
            details={
                "inspected_columns": selected_columns,
                "missing_columns": missing_columns,
            },
        )

    non_numeric_columns = [
        column
        for column in selected_columns
        if not pd.api.types.is_numeric_dtype(dataframe[column])
    ]

    if non_numeric_columns:
        return ValidationResult(
            name="infinite_values",
            passed=False,
            message=(
                "Infinite-value checking requires numeric columns. "
                f"Non-numeric columns: {non_numeric_columns}"
            ),
            details={
                "inspected_columns": selected_columns,
                "non_numeric_columns": non_numeric_columns,
            },
        )

    infinite_counts: dict[str, int] = {}

    for column in selected_columns:
        values = dataframe[column].to_numpy(dtype=float)
        infinite_counts[column] = int(np.isinf(values).sum())

    violating_columns = {
        column: count
        for column, count in infinite_counts.items()
        if count > 0
    }

    passed = len(violating_columns) == 0

    if passed:
        message = "No infinite values were found."
    else:
        message = (
            f"Infinite values were found in "
            f"{len(violating_columns)} column(s)."
        )

    return ValidationResult(
        name="infinite_values",
        passed=passed,
        message=message,
        details={
            "inspected_columns": selected_columns,
            "infinite_counts": infinite_counts,
            "violating_columns": violating_columns,
        },
    )


def check_duplicate_rows(
    dataframe: pd.DataFrame,
    subset: Sequence[str] | None = None,
    max_duplicates: int = 0,
) -> ValidationResult:
    """
    Check the number of duplicated rows beyond their first occurrence.

    Parameters
    ----------
    subset:
        Columns used to define duplicates. When omitted, all columns are used.
    max_duplicates:
        Maximum number of duplicated rows allowed beyond first occurrences.
    """
    _ensure_dataframe(dataframe)

    if max_duplicates < 0:
        raise ValueError("max_duplicates cannot be negative.")

    subset_columns = (
        None
        if subset is None
        else _unique_list(subset)
    )

    if subset_columns is not None:
        missing_columns = _find_missing_columns(
            dataframe,
            subset_columns,
        )

        if missing_columns:
            return ValidationResult(
                name="duplicate_rows",
                passed=False,
                message=(
                    "Duplicate-row check could not inspect absent columns: "
                    f"{missing_columns}"
                ),
                details={
                    "subset": subset_columns,
                    "missing_columns": missing_columns,
                },
            )

    duplicate_mask = dataframe.duplicated(
        subset=subset_columns,
        keep="first",
    )

    duplicate_count = int(duplicate_mask.sum())
    duplicate_indices = [
        int(index)
        if isinstance(index, (int, np.integer))
        else str(index)
        for index in dataframe.index[duplicate_mask].tolist()
    ]

    passed = duplicate_count <= max_duplicates

    if passed:
        message = (
            f"Duplicate count {duplicate_count} is within the allowed "
            f"limit of {max_duplicates}."
        )
    else:
        message = (
            f"Found {duplicate_count} duplicated row(s), exceeding "
            f"the allowed limit of {max_duplicates}."
        )

    return ValidationResult(
        name="duplicate_rows",
        passed=passed,
        message=message,
        details={
            "subset": subset_columns,
            "duplicate_count": duplicate_count,
            "duplicate_indices": duplicate_indices,
            "max_duplicates": max_duplicates,
        },
    )


def check_allowed_values(
    dataframe: pd.DataFrame,
    column: str,
    allowed_values: Iterable[Any],
    allow_missing: bool = False,
) -> ValidationResult:
    """
    Check that a categorical column contains only allowed values.
    """
    _ensure_dataframe(dataframe)

    if column not in dataframe.columns:
        return ValidationResult(
            name=f"allowed_values:{column}",
            passed=False,
            message=f"Column '{column}' is missing.",
            details={
                "column": column,
                "missing_column": True,
            },
        )

    allowed = list(dict.fromkeys(allowed_values))

    observed_non_missing = dataframe[column].dropna().unique().tolist()

    unexpected_values = [
        value
        for value in observed_non_missing
        if value not in allowed
    ]

    missing_count = int(dataframe[column].isna().sum())

    passed = (
        len(unexpected_values) == 0
        and (allow_missing or missing_count == 0)
    )

    if passed:
        message = (
            f"Column '{column}' contains only allowed values."
        )
    else:
        problems = []

        if unexpected_values:
            problems.append(
                f"unexpected values: {unexpected_values}"
            )

        if missing_count > 0 and not allow_missing:
            problems.append(
                f"missing values: {missing_count}"
            )

        message = (
            f"Column '{column}' failed allowed-value validation: "
            + "; ".join(problems)
        )

    return ValidationResult(
        name=f"allowed_values:{column}",
        passed=passed,
        message=message,
        details={
            "column": column,
            "allowed_values": allowed,
            "observed_values": observed_non_missing,
            "unexpected_values": unexpected_values,
            "missing_count": missing_count,
            "allow_missing": allow_missing,
        },
    )


def check_numeric_ranges(
    dataframe: pd.DataFrame,
    ranges: Mapping[
        str,
        tuple[float | None, float | None],
    ],
    inclusive: bool = True,
) -> ValidationResult:
    """
    Check numerical columns against configured minimum and maximum values.

    A bound can be None when that side should not be restricted.

    Example
    -------
    ranges={
        "probability": (0.0, 1.0),
        "temperature": (None, 200.0),
    }
    """
    _ensure_dataframe(dataframe)

    configured_columns = list(ranges.keys())
    missing_columns = _find_missing_columns(
        dataframe,
        configured_columns,
    )

    if missing_columns:
        return ValidationResult(
            name="numeric_ranges",
            passed=False,
            message=(
                "Range check could not inspect absent columns: "
                f"{missing_columns}"
            ),
            details={
                "configured_ranges": dict(ranges),
                "missing_columns": missing_columns,
            },
        )

    non_numeric_columns = [
        column
        for column in configured_columns
        if not pd.api.types.is_numeric_dtype(dataframe[column])
    ]

    if non_numeric_columns:
        return ValidationResult(
            name="numeric_ranges",
            passed=False,
            message=(
                "Range checking requires numeric columns. "
                f"Non-numeric columns: {non_numeric_columns}"
            ),
            details={
                "configured_ranges": dict(ranges),
                "non_numeric_columns": non_numeric_columns,
            },
        )

    violations: dict[str, dict[str, Any]] = {}

    for column, bounds in ranges.items():
        if len(bounds) != 2:
            raise ValueError(
                f"Range for '{column}' must contain two values: "
                "(minimum, maximum)."
            )

        minimum, maximum = bounds

        if (
            minimum is not None
            and maximum is not None
            and minimum > maximum
        ):
            raise ValueError(
                f"Minimum bound exceeds maximum for '{column}'."
            )

        values = dataframe[column]
        valid_values = values.dropna()

        if minimum is None:
            below_mask = pd.Series(
                False,
                index=valid_values.index,
            )
        elif inclusive:
            below_mask = valid_values < minimum
        else:
            below_mask = valid_values <= minimum

        if maximum is None:
            above_mask = pd.Series(
                False,
                index=valid_values.index,
            )
        elif inclusive:
            above_mask = valid_values > maximum
        else:
            above_mask = valid_values >= maximum

        below_count = int(below_mask.sum())
        above_count = int(above_mask.sum())

        if below_count > 0 or above_count > 0:
            violations[column] = {
                "minimum": minimum,
                "maximum": maximum,
                "below_minimum_count": below_count,
                "above_maximum_count": above_count,
                "observed_minimum": (
                    float(valid_values.min())
                    if not valid_values.empty
                    else None
                ),
                "observed_maximum": (
                    float(valid_values.max())
                    if not valid_values.empty
                    else None
                ),
            }

    passed = len(violations) == 0

    if passed:
        message = (
            f"All {len(configured_columns)} configured numeric "
            "range checks passed."
        )
    else:
        message = (
            f"{len(violations)} column(s) contain out-of-range values."
        )

    return ValidationResult(
        name="numeric_ranges",
        passed=passed,
        message=message,
        details={
            "configured_ranges": {
                column: list(bounds)
                for column, bounds in ranges.items()
            },
            "inclusive": inclusive,
            "violations": violations,
        },
    )


def check_class_balance(
    dataframe: pd.DataFrame,
    target_column: str,
    min_classes: int = 2,
    min_samples_per_class: int = 1,
    min_fraction_per_class: float = 0.0,
) -> ValidationResult:
    """
    Check whether the target column has enough classes and samples.

    This does not require a perfectly balanced dataset. It checks only the
    configured minimum requirements.
    """
    _ensure_dataframe(dataframe)

    if min_classes < 1:
        raise ValueError("min_classes must be at least 1.")

    if min_samples_per_class < 1:
        raise ValueError(
            "min_samples_per_class must be at least 1."
        )

    if not 0.0 <= min_fraction_per_class <= 1.0:
        raise ValueError(
            "min_fraction_per_class must be between 0 and 1."
        )

    if target_column not in dataframe.columns:
        return ValidationResult(
            name=f"class_balance:{target_column}",
            passed=False,
            message=f"Target column '{target_column}' is missing.",
            details={
                "target_column": target_column,
                "missing_column": True,
            },
        )

    missing_count = int(
        dataframe[target_column].isna().sum()
    )

    target_values = dataframe[target_column].dropna()
    class_counts_series = target_values.value_counts(
        dropna=False
    )

    class_counts = {
        str(label): int(count)
        for label, count in class_counts_series.items()
    }

    total_non_missing = int(len(target_values))

    if total_non_missing == 0:
        class_fractions: dict[str, float] = {}
    else:
        class_fractions = {
            label: float(count / total_non_missing)
            for label, count in class_counts.items()
        }

    low_count_classes = {
        label: count
        for label, count in class_counts.items()
        if count < min_samples_per_class
    }

    low_fraction_classes = {
        label: fraction
        for label, fraction in class_fractions.items()
        if fraction < min_fraction_per_class
    }

    class_count = len(class_counts)

    passed = (
        class_count >= min_classes
        and len(low_count_classes) == 0
        and len(low_fraction_classes) == 0
    )

    if passed:
        message = (
            f"Target '{target_column}' satisfies the configured "
            "class-balance requirements."
        )
    else:
        problems = []

        if class_count < min_classes:
            problems.append(
                f"found {class_count} class(es), "
                f"required at least {min_classes}"
            )

        if low_count_classes:
            problems.append(
                "classes below minimum sample count: "
                f"{low_count_classes}"
            )

        if low_fraction_classes:
            problems.append(
                "classes below minimum fraction: "
                f"{low_fraction_classes}"
            )

        message = (
            f"Target '{target_column}' failed class-balance "
            "validation: "
            + "; ".join(problems)
        )

    return ValidationResult(
        name=f"class_balance:{target_column}",
        passed=passed,
        message=message,
        details={
            "target_column": target_column,
            "class_counts": class_counts,
            "class_fractions": class_fractions,
            "class_count": class_count,
            "missing_count": missing_count,
            "min_classes": min_classes,
            "min_samples_per_class": min_samples_per_class,
            "min_fraction_per_class": min_fraction_per_class,
            "low_count_classes": low_count_classes,
            "low_fraction_classes": low_fraction_classes,
        },
    )


def validation_passed(
    results: Sequence[ValidationResult],
) -> bool:
    """
    Return True only when every validation result passed.
    """
    return all(result.passed for result in results)


def run_data_checks(
    dataframe: pd.DataFrame,
    *,
    required_columns: Sequence[str] | None = None,
    missing_value_columns: Sequence[str] | None = None,
    max_missing_fraction: float = 0.0,
    infinite_value_columns: Sequence[str] | None = None,
    duplicate_subset: Sequence[str] | None = None,
    max_duplicates: int = 0,
    allowed_values: Mapping[str, Iterable[Any]] | None = None,
    numeric_ranges: Mapping[
        str,
        tuple[float | None, float | None],
    ]
    | None = None,
    target_column: str | None = None,
    min_classes: int = 2,
    min_samples_per_class: int = 1,
    min_fraction_per_class: float = 0.0,
) -> list[ValidationResult]:
    """
    Run a standard collection of dataframe validation checks.

    Missing-value, infinite-value, and duplicate checks always run.
    Other checks run when their corresponding configuration is provided.
    """
    _ensure_dataframe(dataframe)

    results: list[ValidationResult] = []

    if required_columns is not None:
        results.append(
            check_required_columns(
                dataframe,
                required_columns,
            )
        )

    results.append(
        check_missing_values(
            dataframe,
            columns=missing_value_columns,
            max_missing_fraction=max_missing_fraction,
        )
    )

    results.append(
        check_infinite_values(
            dataframe,
            columns=infinite_value_columns,
        )
    )

    results.append(
        check_duplicate_rows(
            dataframe,
            subset=duplicate_subset,
            max_duplicates=max_duplicates,
        )
    )

    if allowed_values is not None:
        for column, values in allowed_values.items():
            results.append(
                check_allowed_values(
                    dataframe,
                    column=column,
                    allowed_values=values,
                )
            )

    if numeric_ranges is not None:
        results.append(
            check_numeric_ranges(
                dataframe,
                ranges=numeric_ranges,
            )
        )

    if target_column is not None:
        results.append(
            check_class_balance(
                dataframe,
                target_column=target_column,
                min_classes=min_classes,
                min_samples_per_class=min_samples_per_class,
                min_fraction_per_class=min_fraction_per_class,
            )
        )

    return results
