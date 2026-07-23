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
)
from ml_validation_toolkit.model_checks import (
    check_confusion_matrix_consistency,
    check_metric_thresholds,
    check_prediction_labels,
    check_prediction_lengths,
    check_probability_matrix,
    check_score_range,
    run_model_checks,
    validation_passed,
)
from ml_validation_toolkit.reporting import (
    ValidationFailureError,
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

__version__ = "0.1.0"

__all__ = [
    "ValidationFailureError",
    "ValidationResult",
    "build_validation_report",
    "check_allowed_values",
    "check_class_balance",
    "check_confusion_matrix_consistency",
    "check_duplicate_rows",
    "check_infinite_values",
    "check_metric_thresholds",
    "check_missing_values",
    "check_numeric_ranges",
    "check_prediction_labels",
    "check_prediction_lengths",
    "check_probability_matrix",
    "check_required_columns",
    "check_score_range",
    "format_validation_summary",
    "print_validation_summary",
    "raise_for_validation_failures",
    "results_to_dataframe",
    "results_to_records",
    "run_data_checks",
    "run_model_checks",
    "save_validation_csv",
    "save_validation_json",
    "summarize_validation_results",
    "validation_passed",
]
