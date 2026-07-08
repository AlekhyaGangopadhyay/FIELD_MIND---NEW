# Blast Vibration Prediction Model Metrics

This document summarizes the performance of the trained machine learning models on the Mount Erzberg production blast vibration dataset. Models were trained using features from trace headers and coordinates merged with aggregated blasthole information from `BLASTS.txt`.

## 1. Classification Models
*Target variable*: **Vibration Hazard** (`vibration_hazard = 1` if Peak Particle Velocity (PPV) > 1.0 mm/s, else `0`)

| Model Name | Algorithm Used | Features Used | Train Accuracy | Test Accuracy | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression (Baseline) | LogisticRegression | Standard (Offset, Charges, Holes, Detonator, Components) | 81.56% | 82.12% | Not Saved |
| Decision Tree Classifier | DecisionTreeClassifier | Standard (Offset, Charges, Holes, Detonator, Components) | 85.66% | 84.78% | Not Saved |
| Random Forest Classifier | RandomForestClassifier | Standard + Spatial Coordinates (Source/Receiver XYZ) | 96.92% | 93.01% | ⭐⭐ **Best (Saved)** |
| Gradient Boosting Classifier | GradientBoostingClassifier | Standard + Spatial + Scaled Distances (USBM, LK, Elevation Diff) | 96.28% | 92.74% | Not Saved |
| Multi-Layer Perceptron (MLP) Classifier | MLPClassifier | Standard + Spatial + Scaled Distances (USBM, LK, Elevation Diff) | 94.90% | 91.73% | Not Saved |

## 2. Regression Models
*Target variable*: **ln(PPV)** (log-transformed Peak Particle Velocity in mm/s)

| Model Name | Algorithm Used | Features Used | Train R2 Score | Test R2 Score | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Ridge Regression (Baseline) | Ridge | Standard (Offset, Charges, Holes, Detonator, Components) | 0.5397 (R2) | 0.5613 (R2) | Not Saved |
| Decision Tree Regressor | DecisionTreeRegressor | Standard (Offset, Charges, Holes, Detonator, Components) | 0.6736 (R2) | 0.6601 (R2) | Not Saved |
| Random Forest Regressor | RandomForestRegressor | Standard + Spatial Coordinates (Source/Receiver XYZ) | 0.9515 (R2) | 0.9019 (R2) | Not Saved |
| Gradient Boosting Regressor | GradientBoostingRegressor | Standard + Spatial + Scaled Distances (USBM, LK, Elevation Diff) | 0.9415 (R2) | 0.9165 (R2) | ⭐⭐ **Best (Saved)** |
| Multi-Layer Perceptron (MLP) Regressor | MLPRegressor | Standard + Spatial + Scaled Distances (USBM, LK, Elevation Diff) | 0.9069 (R2) | 0.8553 (R2) | Not Saved |