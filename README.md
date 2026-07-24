# ML Testing and Validation Toolkit

[![Validation Toolkit Tests](https://github.com/sakthi-kr/ml-testing-validation-toolkit/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/sakthi-kr/ml-testing-validation-toolkit/actions/workflows/tests.yml)

A lightweight Python package for validating machine-learning datasets, predictions, probabilities, metrics, confusion matrices, and generated reports.

The toolkit is used by the accompanying industrial image defect-detection and sensor predictive-maintenance projects. It provides reusable consistency checks; project-specific questions such as leakage, threshold selection, robustness, and external generalization remain the responsibility of each application.

## Main Capabilities

### Data validation

- required-column checks
- missing- and infinite-value checks
- duplicate-row checks
- allowed categorical-value checks
- numerical range checks
- class-representation checks
- combined dataframe-validation workflows

### Model-output validation

- target and prediction length consistency
- allowed prediction-label checks
- probability-matrix shape and row-sum checks
- prediction-score range checks
- metric regression thresholds
- confusion-matrix consistency checks
- combined model-output validation workflows

### Reporting and pipeline control

- readable console summaries
- structured JSON reports
- flat CSV validation tables
- report metadata
- `ValidationFailureError` for failing automated pipelines

## Installation

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The current package version is `0.1.0` and requires Python 3.10 or newer.

## Basic Example

```python
import pandas as pd

from ml_validation_toolkit import (
    print_validation_summary,
    run_data_checks,
    validation_passed,
)

frame = pd.DataFrame(
    {
        "score": [0.1, 0.4, 0.9],
        "label": ["normal", "normal", "defective"],
    }
)

results = run_data_checks(
    frame,
    required_columns=["score", "label"],
    allowed_values={"label": ["normal", "defective"]},
    numeric_ranges={"score": (0.0, 1.0)},
    target_column="label",
)

print_validation_summary(results, report_name="Example Validation")
print("Passed:", validation_passed(results))
```

## Included Examples

### General validation workflow

```bash
python examples/example_validation_workflow.py
```

This combines reusable data and model checks and generates JSON and CSV reports.

### Industrial image anomaly outputs

```bash
python examples/validate_image_anomaly_outputs.py
```

This self-contained example validates:

- image-record schema and duplicate paths
- binary labels and anomaly-score ranges
- binary probability rows
- accuracy, F1, and AUROC thresholds
- confusion-matrix consistency
- JSON and CSV report generation

Outputs are written to:

```text
example_outputs/image_anomaly/
```

### Industrial sensor classification

```bash
python examples/validate_sensor_classification.py
```

This self-contained example validates:

- recording identifiers and operating-load metadata
- sensor-feature ranges
- four fault classes and predicted labels
- four-class probability rows and confidence scores
- accuracy and macro-F1 thresholds
- confusion-matrix consistency
- JSON and CSV report generation

Outputs are written to:

```text
example_outputs/sensor_classification/
```

## Public API

The package exports reusable functions for:

- data checks through `run_data_checks`
- model-output checks through `run_model_checks`
- pass/fail evaluation through `validation_passed`
- console summaries through `print_validation_summary`
- JSON and CSV reporting through `save_validation_json` and `save_validation_csv`
- automated failure handling through `raise_for_validation_failures`

The full exported API is defined in `src/ml_validation_toolkit/__init__.py`.

## Testing and CI

Run the complete suite with:

```bash
pytest
```

GitHub Actions tests the toolkit on Python 3.10 and Python 3.12 for pushes and pull requests to `main`.

The test suite includes end-to-end checks for both industrial examples and verifies their generated JSON and CSV reports.

## Package Structure

```text
ml-testing-validation-toolkit/
├── .github/
│   └── workflows/
│       └── tests.yml
├── examples/
│   ├── __init__.py
│   ├── example_validation_workflow.py
│   ├── validate_image_anomaly_outputs.py
│   └── validate_sensor_classification.py
├── src/
│   └── ml_validation_toolkit/
│       ├── __init__.py
│       ├── data_checks.py
│       ├── model_checks.py
│       └── reporting.py
├── tests/
│   ├── test_project_examples.py
│   └── ...
├── pyproject.toml
└── README.md
```

## Project Integrations

This toolkit is integrated into:

- [`industrial-image-defect-detection`](https://github.com/sakthi-kr/industrial-image-defect-detection)
- [`sensor-predictive-maintenance`](https://github.com/sakthi-kr/sensor-predictive-maintenance)

The application repositories use the toolkit for reusable feature-table, prediction-output, metric, confusion-matrix, generated-report, and path-portability checks. Domain-specific invariants remain inside their respective projects.

## Scope and Limitations

Passing the configured checks confirms that the inspected data and model outputs are internally consistent. It does not by itself prove:

- model generalization
- production readiness
- robustness to distribution changes
- absence of project-specific data leakage
- appropriateness of an operating threshold

Those questions require project-specific experimental design, representative data, and domain validation.
