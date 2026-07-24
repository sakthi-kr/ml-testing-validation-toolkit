"""Validate a small industrial image-anomaly prediction table.

The example is self-contained. It demonstrates how an image anomaly-detection
project can validate prediction rows, anomaly scores, aggregate metrics, and a
confusion matrix before publishing or consuming the outputs downstream.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from ml_validation_toolkit import (
    print_validation_summary,
    raise_for_validation_failures,
    run_data_checks,
    run_model_checks,
    save_validation_csv,
    save_validation_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "example_outputs" / "image_anomaly"


def build_prediction_table() -> pd.DataFrame:
    """Return a deterministic image-level anomaly prediction example."""
    return pd.DataFrame(
        {
            "image_path": [
                "bottle/test/good/001.png",
                "bottle/test/good/002.png",
                "bottle/test/good/003.png",
                "bottle/test/good/004.png",
                "bottle/test/broken_large/001.png",
                "bottle/test/broken_small/001.png",
                "bottle/test/contamination/001.png",
                "bottle/test/contamination/002.png",
            ],
            "category": ["bottle"] * 8,
            "true_label": [0, 0, 0, 0, 1, 1, 1, 1],
            "predicted_label": [0, 0, 0, 1, 1, 1, 0, 1],
            "anomaly_score": [0.05, 0.12, 0.22, 0.56, 0.91, 0.76, 0.48, 0.83],
        }
    )


def run_validation(output_directory: str | Path) -> dict[str, Path]:
    """Run all checks and save JSON and CSV reports."""
    output_directory = Path(output_directory)
    dataframe = build_prediction_table()

    y_true = dataframe["true_label"]
    y_pred = dataframe["predicted_label"]
    scores = dataframe["anomaly_score"]
    probabilities = np.column_stack([1.0 - scores, scores])

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
    }
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    data_results = run_data_checks(
        dataframe,
        required_columns=[
            "image_path",
            "category",
            "true_label",
            "predicted_label",
            "anomaly_score",
        ],
        duplicate_subset=["image_path"],
        allowed_values={
            "category": ["bottle"],
            "true_label": [0, 1],
            "predicted_label": [0, 1],
        },
        numeric_ranges={"anomaly_score": (0.0, 1.0)},
        target_column="true_label",
        min_classes=2,
        min_samples_per_class=4,
    )

    model_results = run_model_checks(
        y_true=y_true,
        y_pred=y_pred,
        allowed_labels=[0, 1],
        probabilities=probabilities,
        expected_probability_classes=2,
        scores=scores,
        score_minimum=0.0,
        score_maximum=1.0,
        metrics=metrics,
        metric_minimums={
            "accuracy": 0.70,
            "f1_score": 0.70,
            "roc_auc": 0.80,
        },
        matrix=matrix,
        confusion_matrix_labels=[0, 1],
    )

    results = [*data_results, *model_results]
    report_name = "Industrial Image Anomaly Output Validation"

    print_validation_summary(results, report_name=report_name)

    json_path = save_validation_json(
        results,
        output_directory / "validation_report.json",
        report_name=report_name,
        metadata={
            "example_type": "image_anomaly_detection",
            "category": "bottle",
            "rows": len(dataframe),
            "metrics": metrics,
            "threshold": 0.5,
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
        description="Validate example image anomaly-detection outputs."
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
