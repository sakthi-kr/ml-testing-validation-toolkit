"""
Reusable data-quality, model-validation, and reporting utilities
for machine-learning projects.
"""

from ml_validation_toolkit.data_checks import (
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

__version__ = "0.1.0"

__all__ = [
    "ValidationResult",
    "check_allowed_values",
    "check_class_balance",
    "check_duplicate_rows",
    "check_infinite_values",
    "check_missing_values",
    "check_numeric_ranges",
    "check_required_columns",
    "run_data_checks",
    "validation_passed",
]
