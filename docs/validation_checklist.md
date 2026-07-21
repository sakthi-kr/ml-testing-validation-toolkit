# ML Validation Checklist

## Data Checks

- [ ] Dataset source is documented
- [ ] Dataset is not committed to GitHub
- [ ] Missing values are checked
- [ ] Feature ranges are checked
- [ ] Train/test split is documented
- [ ] Possible data leakage is discussed

## Model Checks

- [ ] Baseline model is included
- [ ] Evaluation metric is justified
- [ ] Confusion matrix is included
- [ ] False positives are inspected
- [ ] False negatives are inspected
- [ ] Limitations are documented

## Reproducibility

- [ ] Requirements file is included
- [ ] Random seed is set where relevant
- [ ] How-to-run instructions are included
- [ ] Results are saved in `results/` or `reports/`
