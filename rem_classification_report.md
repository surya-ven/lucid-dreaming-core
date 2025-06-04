# Example REM Detection Model - Comprehensive Classification Report

**Generated:** 2025-06-03 13:27:00

## Model Overview

- **Model AUC:** 0.8271
- **Total Epochs:** 1000
- **REM Epochs (Class 1):** 243 (24.3%)
- **Non-REM Epochs (Class 0):** 757 (75.7%)
- **Score Range:** [0.0054, 0.9627]

## Summary Metrics Table

| Threshold Type | Threshold | Sensitivity | Specificity | Precision | Accuracy | FPR |
|---|---|---|---|---|---|---|
| Conservative (FPR ≤ 5%) | 0.7786 | 0.066 | 0.999 | 0.941 | 0.772 | 0.001 |
| Balanced (FPR ≤ 10%) | 0.5449 | 0.399 | 0.923 | 0.626 | 0.796 | 0.077 |
| Sensitive (70% confidence) | 0.4382 | 0.658 | 0.802 | 0.516 | 0.767 | 0.198 |
| High Confidence (80%) | 0.6500 | 0.239 | 0.982 | 0.806 | 0.801 | 0.018 |
| Very High Confidence (90%) | 0.7400 | 0.115 | 0.996 | 0.903 | 0.782 | 0.004 |

## Confusion Matrix Table

| Threshold Type | TP | TN | FP | FN | Total Pos | Total Neg |
|---|---|---|---|---|---|---|
| Conservative (FPR ≤ 5%) | 16 | 756 | 1 | 227 | 17 | 983 |
| Balanced (FPR ≤ 10%) | 97 | 699 | 58 | 146 | 155 | 845 |
| Sensitive (70% confidence) | 160 | 607 | 150 | 83 | 310 | 690 |
| High Confidence (80%) | 58 | 743 | 14 | 185 | 72 | 928 |
| Very High Confidence (90%) | 28 | 754 | 3 | 215 | 31 | 969 |

## Detailed Classification Reports

### Conservative (FPR ≤ 5%) (Threshold = 0.7786)

```
              precision    recall  f1-score   support

     Non-REM     0.7691    0.9987    0.8690       757
         REM     0.9412    0.0658    0.1231       243

    accuracy                         0.7720      1000
   macro avg     0.8551    0.5323    0.4960      1000
weighted avg     0.8109    0.7720    0.6877      1000
```

### Balanced (FPR ≤ 10%) (Threshold = 0.5449)

```
              precision    recall  f1-score   support

     Non-REM     0.8272    0.9234    0.8727       757
         REM     0.6258    0.3992    0.4874       243

    accuracy                         0.7960      1000
   macro avg     0.7265    0.6613    0.6800      1000
weighted avg     0.7783    0.7960    0.7791      1000
```

### Sensitive (70% confidence) (Threshold = 0.4382)

```
              precision    recall  f1-score   support

     Non-REM     0.8797    0.8018    0.8390       757
         REM     0.5161    0.6584    0.5787       243

    accuracy                         0.7670      1000
   macro avg     0.6979    0.7301    0.7088      1000
weighted avg     0.7914    0.7670    0.7757      1000
```

### High Confidence (80%) (Threshold = 0.6500)

```
              precision    recall  f1-score   support

     Non-REM     0.8006    0.9815    0.8819       757
         REM     0.8056    0.2387    0.3683       243

    accuracy                         0.8010      1000
   macro avg     0.8031    0.6101    0.6251      1000
weighted avg     0.8018    0.8010    0.7571      1000
```

### Very High Confidence (90%) (Threshold = 0.7400)

```
              precision    recall  f1-score   support

     Non-REM     0.7781    0.9960    0.8737       757
         REM     0.9032    0.1152    0.2044       243

    accuracy                         0.7820      1000
   macro avg     0.8407    0.5556    0.5390      1000
weighted avg     0.8085    0.7820    0.7111      1000
```

## Threshold Strategy Comparison

### FPR-Based vs Confidence-Based Thresholds

**Conservative Strategy (FPR ≤ 5%):**
- **Goal:** Minimize false alarms from Non-REM epochs
- **Result:** Only 0.1% of Non-REM epochs incorrectly classified as REM
- **Trade-off:** Lower sensitivity (6.6%) - only catches 6.6% of actual REM epochs

**High Confidence Strategy (90% confidence):**
- **Goal:** High confidence when predicting REM
- **Result:** When predicting REM, correct 90.3% of the time
- **Trade-off:** Lower sensitivity - only catches a subset of actual REM epochs

### Recommendations

**For Clinical Research (High Precision Required):**
- Use **Conservative (FPR ≤ 5%)** threshold (0.7786)
- Achieves 94.1% precision with 6.6% sensitivity

**For Real-time Lucid Dream Detection (High Sensitivity Required):**
- Use **Sensitive (70% confidence)** threshold (0.4382)
- Achieves 65.8% sensitivity with 51.6% precision

**For Balanced Performance:**
- Use **Balanced (FPR ≤ 10%)** threshold (0.5449)
- Achieves 39.9% sensitivity with 62.6% precision

## REM Detection Context

REM sleep detection has specific requirements compared to general classification tasks:

- **REM constitutes ~20-25% of total sleep** - Class imbalance is expected
- **False positives can disrupt sleep** - Conservative thresholds preferred for sleep applications
- **Missing REM periods reduces study validity** - Balance needed between precision and recall
- **Real-time detection requires fast response** - Consider latency vs accuracy trade-offs

## Metrics Legend

- **Sens (Sensitivity):** True Positive Rate (Recall) - Proportion of actual REM epochs correctly identified
- **Spec (Specificity):** True Negative Rate - Proportion of actual Non-REM epochs correctly identified  
- **Prec (Precision):** Positive Predictive Value - Proportion of REM predictions that are correct
- **Acc (Accuracy):** Overall proportion of correct predictions
- **FPR:** False Positive Rate (1 - Specificity) - Proportion of Non-REM epochs incorrectly classified as REM
- **TP:** True Positives - Correctly identified REM epochs
- **TN:** True Negatives - Correctly identified Non-REM epochs
- **FP:** False Positives - Non-REM epochs incorrectly classified as REM
- **FN:** False Negatives - REM epochs incorrectly classified as Non-REM

---
*Report generated by REM Detection Classification Analysis Script*
