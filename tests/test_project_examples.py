"""End-to-end tests for the two industrial validation examples."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import ml_validation_toolkit
from examples.validate_image_anomaly_outputs import (
    run_validation as run_image_validation,
)
from examples.validate_sensor_classification import (
    run_validation as run_sensor_validation,
)


def assert_valid_example_outputs(paths: dict[str, Path]) -> dict:
    assert set(paths) == {"json", "csv"}
    assert paths["json"].is_file()
    assert paths["csv"].is_file()

    report = json.loads(paths["json"].read_text(encoding="utf-8"))
    checks = pd.read_csv(paths["csv"])

    assert report["schema_version"] == "1.0"
    assert report["summary"]["overall_passed"] is True
    assert report["summary"]["failed_checks"] == 0
    assert len(checks) == report["summary"]["total_checks"]
    assert checks["passed"].all()
    return report


def test_public_api_exposes_core_workflow_functions() -> None:
    expected_names = {
        "run_data_checks",
        "run_model_checks",
        "save_validation_json",
        "save_validation_csv",
        "raise_for_validation_failures",
        "validation_passed",
    }

    assert ml_validation_toolkit.__version__ == "0.1.0"
    assert expected_names.issubset(set(ml_validation_toolkit.__all__))


def test_image_anomaly_example_generates_passing_reports(tmp_path: Path) -> None:
    report = assert_valid_example_outputs(
        run_image_validation(tmp_path / "image")
    )

    assert report["report_name"] == "Industrial Image Anomaly Output Validation"
    assert report["metadata"]["example_type"] == "image_anomaly_detection"
    assert report["metadata"]["category"] == "bottle"
    assert report["metadata"]["rows"] == 8


def test_sensor_example_generates_passing_reports(tmp_path: Path) -> None:
    report = assert_valid_example_outputs(
        run_sensor_validation(tmp_path / "sensor")
    )

    assert report["report_name"] == "Industrial Sensor Classification Validation"
    assert report["metadata"]["example_type"] == "sensor_fault_classification"
    assert report["metadata"]["rows"] == 8
    assert len(report["metadata"]["classes"]) == 4


def test_example_reports_use_different_domain_metadata(tmp_path: Path) -> None:
    image_report = json.loads(
        run_image_validation(tmp_path / "image")["json"].read_text(
            encoding="utf-8"
        )
    )
    sensor_report = json.loads(
        run_sensor_validation(tmp_path / "sensor")["json"].read_text(
            encoding="utf-8"
        )
    )

    assert image_report["metadata"]["example_type"] != sensor_report["metadata"][
        "example_type"
    ]
    assert "threshold" in image_report["metadata"]
    assert "operating_loads_hp" in sensor_report["metadata"]
