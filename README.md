# ML Testing and Validation Toolkit

[![Validation Toolkit Tests](https://github.com/sakthi-kr/ml-testing-validation-toolkit/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/sakthi-kr/ml-testing-validation-toolkit/actions/workflows/tests.yml)

A reusable Python package for validating machine-learning datasets, predictions,
probabilities, metrics, confusion matrices, and generated validation reports.
It is designed for small and medium ML projects where data and model outputs
should be checked before results are published or consumed by another pipeline.

The toolkit is used by two accompanying industrial ML projects:

- [Sensor Predictive Maintenance](https://github.com/sakthi-kr/sensor-predictive-maintenance)
- [Industrial Image Defect Detection](https://github.com/sakthi-kr/industrial-image-defect-detection)

## What the toolkit validates

### Dataset checks

- required columns
- missing-value fractions
- infinite values
- duplicate rows or identifiers
- allowed categorical values
- configured numerical ranges
- minimum class representation
- combined dataframe-validation workflows

### Model-output checks

- matching target, prediction, score, and probability lengths
- allowed target and prediction labels
- probability-matrix dimensions, finite values, bounds, and row sums
- score ranges
- minimum or maximum metric thresholds
- confusion-matrix consistency
- combined model-output validation workflows

### Reporting and pipeline control

- readable console summaries
- strict JSON reports with metadata
- flat CSV check tables
- an exception that can stop CI or a pipeline when validation fails

## Installation

The project uses Python 3.10 or newer.

### Editable development installation

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On PowerShell, activate with:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Use from another local portfolio project

From the consuming project directory:

```bash
python -m pip install -e ../ml-testing-validation-toolkit
```

The editable installation keeps the application project connected to the local
toolkit source while the portfolio is under development.

## Quick start

```python
import pandas as pd

from ml_validation_toolkit import (
    print_validation_summary,
    raise_for_validation_failures,
    run_data_checks,
    save_validation_json,
)

features = pd.DataFrame(
    {
        "sample_id": ["a", "b", "c"],
        "rms": [0.2, 0.5, 0.8],
        "label": ["normal", "fault", "fault"],
    }
)

results = run_data_checks(
    features,
    required_columns=["sample_id", "rms", "label"],
    duplicate_subset=["sample_id"],
    allowed_values={"label": ["normal", "fault"]},
    numeric_ranges={"rms": (0.0, 2.0)},
    target_column="label",
)

print_validation_summary(results, report_name="Feature Table Validation")
save_validation_json(
    results,
    "validation_report.json",
    report_name="Feature Table Validation",
)
raise_for_validation_failures(results)
```

Ordinary failed checks return structured `ValidationResult` objects. Calling
`raise_for_validation_failures` is optional and is intended for automated
pipelines where a failed check should stop execution.

## Industrial examples

The repository includes three self-contained examples.

### General validation workflow

```bash
python examples/example_validation_workflow.py
```

### Image anomaly-detection outputs

Validates image paths, class labels, anomaly scores, probabilities, aggregate
metrics, and a confusion matrix:

```bash
python examples/validate_image_anomaly_outputs.py
```

### Sensor fault-classification outputs

Validates recording identifiers, operating loads, feature ranges, four-class
predictions, probability rows, aggregate metrics, and a confusion matrix:

```bash
python examples/validate_sensor_classification.py
```

Each example writes a JSON report and CSV check table under `example_outputs/`.
A custom output directory can be supplied to the two industrial examples:

```bash
python examples/validate_image_anomaly_outputs.py --output-dir reports/image
python examples/validate_sensor_classification.py --output-dir reports/sensor
```

The examples use deterministic in-memory data. They demonstrate the validation
contract without requiring the CWRU or MVTec AD datasets.

## Public API overview

| Area | Main functions |
|---|---|
| Data validation | `run_data_checks`, `check_required_columns`, `check_missing_values`, `check_infinite_values`, `check_duplicate_rows`, `check_allowed_values`, `check_numeric_ranges`, `check_class_balance` |
| Model validation | `run_model_checks`, `check_prediction_lengths`, `check_prediction_labels`, `check_probability_matrix`, `check_score_range`, `check_metric_thresholds`, `check_confusion_matrix_consistency` |
| Reporting | `print_validation_summary`, `save_validation_json`, `save_validation_csv`, `build_validation_report`, `summarize_validation_results` |
| Pipeline control | `validation_passed`, `raise_for_validation_failures`, `ValidationFailureError` |

The package version is available as:

```python
import ml_validation_toolkit

print(ml_validation_toolkit.__version__)
```

## Testing and continuous integration

Run the complete test suite:

```bash
python -m pytest
```

Run the two industrial example tests only:

```bash
python -m pytest -q tests/test_project_examples.py
```

GitHub Actions installs the package and runs compilation and tests on Python
3.10 and Python 3.12 for pushes and pull requests to `main`.

## Documentation assets

- [`docs/validation_checklist.md`](docs/validation_checklist.md) — project-level ML validation checklist
- [`docs/model_card_template.md`](docs/model_card_template.md) — compact model-card structure

## Repository structure

```text
ml-testing-validation-toolkit/
├── .github/
│   └── workflows/
│       └── tests.yml
├── docs/
│   ├── model_card_template.md
│   └── validation_checklist.md
├── examples/
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
├── pyproject.toml
└── README.md
```

## Scope and limitations

Passing these checks means the configured inputs and outputs satisfy the stated
validation rules. It does **not** by itself prove:

- model generalization to new machines, products, or operating conditions
- absence of train/test leakage
- robustness to distribution shift
- production readiness
- calibration of probabilities or anomaly scores
- suitability of a classification or review threshold
- domain validity of the selected metric limits

Those questions require project-specific experimental design, grouped or
time-aware evaluation where appropriate, failure analysis, and domain review.

## Current status

Version `0.1.0` provides a tested public API for dataframe checks, model-output
checks, structured reporting, and CI-friendly validation failures. The toolkit
is intentionally lightweight and is not published to PyPI; it is installed
locally in editable mode by the two portfolio application repositories.
