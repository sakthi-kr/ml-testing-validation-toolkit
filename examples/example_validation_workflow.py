"""
Example combining reusable data and model validation checks.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ml_validation_toolkit import (
    print_validation_summary,
    raise_for_validation_failures,
    run_data_checks,
    run_model_checks,
    save_validation_csv,
    save_validation_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = PROJECT_ROOT / "example_outputs"


def main() -> None:
    dataframe = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "sensor_value": [0.1, 0.4, 0.8, 0.9],
            "true_label": [0, 0, 1, 1],
            "predicted_label": [0, 0, 1, 1],
            "defect_probability": [0.05, 0.20, 0.82, 0.91],
        }
    )

    probabilities = np.column_stack(
        [
            1.0 - dataframe["defect_probability"],
            dataframe["defect_probability"],
        ]
    )

    data_results = run_data_checks(
        dataframe,
        required_columns=[
            "sample_id",
            "sensor_value",
            "true_label",
            "predicted_label",
            "defect_probability",
        ],
        duplicate_subset=["sample_id"],
        allowed_values={
            "true_label": [0, 1],
            "predicted_label": [0, 1],
        },
        numeric_ranges={
            "sensor_value": (0.0, 1.0),
            "defect_probability": (0.0, 1.0),
        },
        target_column="true_label",
        min_classes=2,
        min_samples_per_class=2,
    )

    model_results = run_model_checks(
        y_true=dataframe["true_label"],
        y_pred=dataframe["predicted_label"],
        allowed_labels=[0, 1],
        probabilities=probabilities,
        expected_probability_classes=2,
        scores=dataframe["defect_probability"],
        score_minimum=0.0,
        score_maximum=1.0,
        metrics={
            "accuracy": 1.0,
            "f1_score": 1.0,
        },
        metric_minimums={
            "accuracy": 0.90,
            "f1_score": 0.90,
        },
        matrix=[
            [2, 0],
            [0, 2],
        ],
        confusion_matrix_labels=[0, 1],
    )

    all_results = [
        *data_results,
        *model_results,
    ]

    print_validation_summary(
        all_results,
        report_name="Example ML Validation Report",
    )

    json_path = save_validation_json(
        all_results,
        OUTPUT_DIRECTORY / "validation_report.json",
        report_name="Example ML Validation Report",
        metadata={
            "model": "example-binary-classifier",
            "dataset_rows": len(dataframe),
            "purpose": "toolkit demonstration",
        },
    )

    csv_path = save_validation_csv(
        all_results,
        OUTPUT_DIRECTORY / "validation_checks.csv",
    )

    print("\nSaved outputs")
    print("-------------")
    print(f"JSON report: {json_path}")
    print(f"CSV report : {csv_path}")

    raise_for_validation_failures(
        all_results
    )


if __name__ == "__main__":
    main()
