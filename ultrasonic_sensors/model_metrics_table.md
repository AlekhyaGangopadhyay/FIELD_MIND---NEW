# Robot Navigation Ultrasonic Sensor Classification Metrics

This document summarizes the performance of various machine learning models trained on the SCITOS-G5 robot navigation dataset using 2, 4, and 24 ultrasonic sensor configurations.

The target variable is the robot navigation decision Class: `Move-Forward`, `Slight-Right-Turn`, `Sharp-Right-Turn`, or `Slight-Left-Turn`.


## 2-sensor configuration
| Model Name | Algorithm Used | Train Accuracy | Test Accuracy | Status |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression (Baseline) | LogisticRegression | 94.13% | 94.32% | Not Saved |
| Decision Tree Classifier | DecisionTreeClassifier | 100.00% | 100.00% | ⭐⭐ **Best (Saved)** |
| Random Forest Classifier | RandomForestClassifier | 100.00% | 100.00% | Not Saved |
| Gradient Boosting Classifier | GradientBoostingClassifier | 100.00% | 100.00% | Not Saved |
| Multi-Layer Perceptron (MLP) Classifier | MLPClassifier | 99.56% | 99.54% | Not Saved |

## 4-sensor configuration
| Model Name | Algorithm Used | Train Accuracy | Test Accuracy | Status |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression (Baseline) | LogisticRegression | 94.20% | 94.60% | Not Saved |
| Decision Tree Classifier | DecisionTreeClassifier | 100.00% | 100.00% | ⭐⭐ **Best (Saved)** |
| Random Forest Classifier | RandomForestClassifier | 100.00% | 99.82% | Not Saved |
| Gradient Boosting Classifier | GradientBoostingClassifier | 100.00% | 100.00% | Not Saved |
| Multi-Layer Perceptron (MLP) Classifier | MLPClassifier | 99.50% | 99.54% | Not Saved |

## 24-sensor configuration
| Model Name | Algorithm Used | Train Accuracy | Test Accuracy | Status |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression (Baseline) | LogisticRegression | 71.29% | 69.23% | Not Saved |
| Decision Tree Classifier | DecisionTreeClassifier | 100.00% | 99.27% | Not Saved |
| Random Forest Classifier | RandomForestClassifier | 100.00% | 99.36% | Not Saved |
| Gradient Boosting Classifier | GradientBoostingClassifier | 100.00% | 99.54% | ⭐⭐ **Best (Saved)** |
| Multi-Layer Perceptron (MLP) Classifier | MLPClassifier | 99.50% | 92.22% | Not Saved |