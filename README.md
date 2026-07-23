# ML Testing and Validation Toolkit

[![Validation Toolkit Tests](https://github.com/sakthi-kr/ml-testing-validation-toolkit/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/sakthi-kr/ml-testing-validation-toolkit/actions/workflows/tests.yml)

## Summary

A reusable Python toolkit for validating machine-learning datasets, predictions, probabilities, metrics, confusion matrices, and generated model reports.

The package is used by the accompanying sensor predictive-maintenance and industrial image defect-detection projects.

## Main Capabilities

### Data Validation

- required-column checks
- missing-value checks
- infinite-value checks
- duplicate-row checks
- allowed categorical-value checks
- numerical range checks
- class-representation checks
- combined dataframe-validation workflows

### Model-Output Validation

- target and prediction length consistency
- allowed prediction-label checks
- probability-matrix shape and row-sum checks
- prediction-score range checks
- metric regression thresholds
- confusion-matrix consistency checks
- combined model-output validation workflows

### Reporting

- readable console summaries
- structured JSON reports
- flat CSV validation tables
- metadata support
- automated pipeline exceptions when validation fails

## Installation

Create and activate a Python environment, then install the package in editable mode:

```bash
python -m venv .venv
source .venv/Scripts/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Basic Example

```python
import pandas as pd

from ml_validation_toolkit import (
    print_validation_summary,
    run_data_checks,
    save_validation_json,
    validation_passed,
)

dataframe = pd.DataFrame(
    {
        "score": [0.1, 0.4, 0.9],
        "label": ["normal", "normal", "defective"],
    }
)

results = run_data_checks(
    dataframe,
    required_columns=["score", "label"],
    allowed_values={
        "label": ["normal", "defective"],
    },
    numeric_ranges={
        "score": (0.0, 1.0),
    },
    target_column="label",
)

print_validation_summary(
    results,
    report_name="Example Validation",
)

save_validation_json(
    results,
    "validation_report.json",
    report_name="Example Validation",
)

print("Passed:", validation_passed(results))
```

## Example Workflow

Run the included demonstration:

```bash
python examples/example_validation_workflow.py
```

The example combines data and model checks and generates JSON and CSV reports.

## Testing

Run the complete test suite:

```bash
pytest
```

The repository also uses GitHub Actions to run the tests automatically on Python 3.10 and Python 3.12 for pushes and pull requests to `main`.

## Package Structure

```text
ml-testing-validation-toolkit/
├── .github/
│   └── workflows/
│       └── tests.yml
├── examples/
│   └── example_validation_workflow.py
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

## Project Integrations

This toolkit is integrated into:

- `sensor-predictive-maintenance`
- `industrial-image-defect-detection`

These integrations validate real feature tables, prediction outputs, metrics, confusion matrices, generated reports, and output-path portability.

## Scope and Limitations

Passing the toolkit checks confirms that configured data and model outputs are internally consistent.

It does not by itself prove:

- model generalization
- production readiness
- robustness to distribution changes
- absence of data leakage
- appropriate operating thresholds

Those questions require project-specific experimental design and domain validation.
