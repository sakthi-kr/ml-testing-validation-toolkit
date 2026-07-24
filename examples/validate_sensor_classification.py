"""Validate a small industrial sensor-classification result table.

The example is self-contained. It shows how a predictive-maintenance workflow
can validate feature ranges, operating-condition metadata, multi-class labels,
probabilities, aggregate metrics, and a confusion matrix.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from ml_validation_toolkit import (
    print_validation_summary,
    raise_for_validation_failures,
    run_data_checks,
    run_model_checks,
    save_validation_csv,
    save_validation_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "example_outputs" / "sensor_classification"
LABELS = ["normal", "ball_fault", "inner_race_fault", "outer_race_fault"]


def build_prediction_table() -> tuple[pd.DataFrame, np.ndarray]:
    """Return deterministic sensor features, predictions, and probabilities."""
    true_labels = [
        "normal",
        "normal",
        "ball_fault",
        "ball_fault",
        "inner_race_fault",
        "inner_race_fault",
        "outer_race_fault",
        "outer_race_fault",
    ]
    predicted_labels = [
        "normal",
        "normal",
        "ball_fault",
        "ball_fault",
        "inner_race_fault",
        "outer_race_fault",
        "outer_race_fault",
        "outer_race_fault",
    ]

    dataframe = pd.DataFrame(
        {
            "recording_id": [f"recording_{index:02d}" for index in range(1, 9)],
            "load_hp": [0, 3, 0, 3, 0, 3, 0, 3],
            "rms": [0.18, 0.21, 0.55, 0.61, 0.74, 0.79, 0.68, 0.72],
            "kurtosis": [2.8, 3.1, 5.9, 6.4, 7.2, 7.7, 6.8, 7.0],
            "true_label": true_labels,
            "predicted_label": predicted_labels,
        }
    )

    probabilities = np.asarray(
        [
            [0.88, 0.05, 0.04, 0.03],
            [0.81, 0.07, 0.07, 0.05],
            [0.04, 0.86, 0.05, 0.05],
            [0.05, 0.79, 0.08, 0.08],
            [0.03, 0.06, 0.82, 0.09],
            [0.04, 0.07, 0.41, 0.48],
            [0.03, 0.05, 0.08, 0.84],
            [0.04, 0.06, 0.10, 0.80],
        ],
        dtype=float,
    )

    return dataframe, probabilities


def run_validation(output_directory: str | Path) -> dict[str, Path]:
    """Run all checks and save JSON and CSV reports."""
    output_directory = Path(output_directory)
    dataframe, probabilities = build_prediction_table()

    y_true = dataframe["true_label"]
    y_pred = dataframe["predicted_label"]
    confidence = probabilities.max(axis=1)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)

    data_results = run_data_checks(
        dataframe,
        required_columns=[
            "recording_id",
            "load_hp",
            "rms",
            "kurtosis",
            "true_label",
            "predicted_label",
        ],
        duplicate_subset=["recording_id"],
        allowed_values={
            "true_label": LABELS,
            "predicted_label": LABELS,
        },
        numeric_ranges={
            "load_hp": (0.0, 3.0),
            "rms": (0.0, 2.0),
            "kurtosis": (0.0, 20.0),
        },
        target_column="true_label",
        min_classes=4,
        min_samples_per_class=2,
    )

    model_results = run_model_checks(
        y_true=y_true,
        y_pred=y_pred,
        allowed_labels=LABELS,
        probabilities=probabilities,
        expected_probability_classes=4,
        scores=confidence,
        score_minimum=0.0,
        score_maximum=1.0,
        metrics=metrics,
        metric_minimums={
            "accuracy": 0.75,
            "macro_f1": 0.70,
        },
        matrix=matrix,
        confusion_matrix_labels=LABELS,
    )

    results = [*data_results, *model_results]
    report_name = "Industrial Sensor Classification Validation"

    print_validation_summary(results, report_name=report_name)

    json_path = save_validation_json(
        results,
        output_directory / "validation_report.json",
        report_name=report_name,
        metadata={
            "example_type": "sensor_fault_classification",
            "classes": LABELS,
            "rows": len(dataframe),
            "metrics": metrics,
            "operating_loads_hp": sorted(dataframe["load_hp"].unique().tolist()),
        },
    )
    csv_path = save_validation_csv(
        results,
        output_directory / "validation_checks.csv",
    )

    raise_for_validation_failures(results)
    return {"json": json_path, "csv": csv_path}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate example sensor fault-classification outputs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for the generated JSON and CSV validation reports.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    paths = run_validation(arguments.output_dir)
    print("\nSaved outputs")
    print("-------------")
    print(f"JSON report: {paths['json']}")
    print(f"CSV report : {paths['csv']}")


if __name__ == "__main__":
    main()
