# Conv1D Model - Comprehensive Classification Report

**Generated:** 2025-06-03 12:25:55
**Details:** YASA sleep stage model adapted and applied to 20 nights of frenz data as binary REM classifier.


## Model Overview

- **Model AUC:** 0.8600
- **Total Samples:** 319
- **LRLR Samples (Class 1):** 85 (26.6%)
- **Non-LRLR Samples (Class 0):** 234 (73.4%)
- **Score Range:** [0.1576, 0.6838]

## Summary Metrics Table

| Threshold Type | Threshold | Sensitivity | Specificity | Precision | Accuracy | FPR |
|---|---|---|---|---|---|---|
| Best AUC (Youden's J) | 0.4479 | 0.800 | 0.774 | 0.562 | 0.781 | 0.226 |
| FPR ≤ 0.05 | 0.4951 | 0.259 | 0.991 | 0.917 | 0.796 | 0.009 |
| FPR ≤ 0.1 | 0.4864 | 0.612 | 0.915 | 0.722 | 0.834 | 0.085 |
| Precision ≥ 0.9 | 0.5000 | 0.247 | 0.996 | 0.955 | 0.796 | 0.004 |
| Precision ≥ 0.95 | 0.5000 | 0.247 | 0.996 | 0.955 | 0.796 | 0.004 |

## Confusion Matrix Table

| Threshold Type | TP | TN | FP | FN | Total Pos | Total Neg |
|---|---|---|---|---|---|---|
| Best AUC (Youden's J) | 68 | 181 | 53 | 17 | 121 | 198 |
| FPR ≤ 0.05 | 22 | 232 | 2 | 63 | 24 | 295 |
| FPR ≤ 0.1 | 52 | 214 | 20 | 33 | 72 | 247 |
| Precision ≥ 0.9 | 21 | 233 | 1 | 64 | 22 | 297 |
| Precision ≥ 0.95 | 21 | 233 | 1 | 64 | 22 | 297 |

## Detailed Classification Reports

### Best AUC (Youden's J) (Threshold = 0.4479)

```
              precision    recall  f1-score   support

    Non-LRLR     0.9141    0.7735    0.8380       234
        LRLR     0.5620    0.8000    0.6602        85

    accuracy                         0.7806       319
   macro avg     0.7381    0.7868    0.7491       319
weighted avg     0.8203    0.7806    0.7906       319
```

### FPR ≤ 0.05 (Threshold = 0.4951)

```
              precision    recall  f1-score   support

    Non-LRLR     0.7864    0.9915    0.8771       234
        LRLR     0.9167    0.2588    0.4037        85

    accuracy                         0.7962       319
   macro avg     0.8516    0.6251    0.6404       319
weighted avg     0.8211    0.7962    0.7510       319
```

### FPR ≤ 0.1 (Threshold = 0.4864)

```
              precision    recall  f1-score   support

    Non-LRLR     0.8664    0.9145    0.8898       234
        LRLR     0.7222    0.6118    0.6624        85

    accuracy                         0.8339       319
   macro avg     0.7943    0.7631    0.7761       319
weighted avg     0.8280    0.8339    0.8292       319
```

### Precision ≥ 0.9 (Threshold = 0.5000)

```
              precision    recall  f1-score   support

    Non-LRLR     0.7845    0.9957    0.8776       234
        LRLR     0.9545    0.2471    0.3925        85

    accuracy                         0.7962       319
   macro avg     0.8695    0.6214    0.6351       319
weighted avg     0.8298    0.7962    0.7483       319
```

### Precision ≥ 0.95 (Threshold = 0.5000)

```
              precision    recall  f1-score   support

    Non-LRLR     0.7845    0.9957    0.8776       234
        LRLR     0.9545    0.2471    0.3925        85

    accuracy                         0.7962       319
   macro avg     0.8695    0.6214    0.6351       319
weighted avg     0.8298    0.7962    0.7483       319
```

## Threshold Strategy Comparison

### FPR-Based vs Precision-Based Thresholds

**FPR ≤ 0.1 Strategy:**
- **Goal:** Minimize false alarms from Non-LRLR samples
- **Result:** Only 8.5% of Non-LRLR samples incorrectly classified as LRLR
- **Trade-off:** Lower precision (72.2%) - when predicting LRLR, only correct 72.2% of the time

**Precision ≥ 0.9 Strategy:**
- **Goal:** High confidence when predicting LRLR
- **Result:** When predicting LRLR, correct 95.5% of the time
- **Trade-off:** Lower sensitivity (24.7%) - only catches 24.7% of actual LRLR events

### Recommendations

**For Diagnosis Environments (High Precision Required):**
- Use **Precision ≥ 0.9** threshold (0.5000)
- Achieves 95.5% precision with 24.7% sensitivity

**For Screening Environments (High Sensitivity Required):**
- Use **Best AUC** threshold (0.4479)
- Achieves 80.0% sensitivity with 56.2% precision

**For Balanced Performance:**
- Use **FPR ≤ 0.1** threshold (0.4864)
- Achieves 61.2% sensitivity with 72.2% precision

## Metrics Legend

- **Sens (Sensitivity):** True Positive Rate (Recall) - Proportion of actual LRLR events correctly identified
- **Spec (Specificity):** True Negative Rate - Proportion of actual Non-LRLR events correctly identified
- **Prec (Precision):** Positive Predictive Value - Proportion of LRLR predictions that are correct
- **Acc (Accuracy):** Overall proportion of correct predictions
- **FPR:** False Positive Rate (1 - Specificity) - Proportion of Non-LRLR events incorrectly classified as LRLR
- **TP:** True Positives - Correctly identified LRLR events
- **TN:** True Negatives - Correctly identified Non-LRLR events
- **FP:** False Positives - Non-LRLR events incorrectly classified as LRLR
- **FN:** False Negatives - LRLR events incorrectly classified as Non-LRLR

---
*Report generated by Conv1D Model evaluation script*
