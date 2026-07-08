# Robot Navigation Ultrasonic Sensor Classification Metrics

This document summarizes the performance of various machine learning models trained on the SCITOS-G5 robot navigation dataset.

| Model Name | Algo Used | Dataset Used | Training Accuracy | Features Taken | Testing Accuracy | Remarks |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression (Baseline) | LogisticRegression | sensor_readings_2.csv | 94.13% | SD_front, SD_left | 94.32% | Not Saved |
| Decision Tree Classifier | DecisionTreeClassifier | sensor_readings_2.csv | 100.00% | SD_front, SD_left | 100.00% | ⭐⭐ **Best (Saved)** |
| Random Forest Classifier | RandomForestClassifier | sensor_readings_2.csv | 100.00% | SD_front, SD_left | 100.00% | Not Saved |
| Gradient Boosting Classifier | GradientBoostingClassifier | sensor_readings_2.csv | 100.00% | SD_front, SD_left | 100.00% | Not Saved |
| Multi-Layer Perceptron (MLP) Classifier | MLPClassifier | sensor_readings_2.csv | 99.56% | SD_front, SD_left | 99.54% | Not Saved |
| Logistic Regression (Baseline) | LogisticRegression | sensor_readings_4.csv | 94.20% | SD_front, SD_left, SD_right, SD_back | 94.60% | Not Saved |
| Decision Tree Classifier | DecisionTreeClassifier | sensor_readings_4.csv | 100.00% | SD_front, SD_left, SD_right, SD_back | 100.00% | ⭐⭐ **Best (Saved)** |
| Random Forest Classifier | RandomForestClassifier | sensor_readings_4.csv | 100.00% | SD_front, SD_left, SD_right, SD_back | 99.82% | Not Saved |
| Gradient Boosting Classifier | GradientBoostingClassifier | sensor_readings_4.csv | 100.00% | SD_front, SD_left, SD_right, SD_back | 100.00% | Not Saved |
| Multi-Layer Perceptron (MLP) Classifier | MLPClassifier | sensor_readings_4.csv | 99.50% | SD_front, SD_left, SD_right, SD_back | 99.54% | Not Saved |
| Logistic Regression (Baseline) | LogisticRegression | sensor_readings_24.csv | 71.29% | US1 to US24 | 69.23% | Not Saved |
| Decision Tree Classifier | DecisionTreeClassifier | sensor_readings_24.csv | 100.00% | US1 to US24 | 99.27% | Not Saved |
| Random Forest Classifier | RandomForestClassifier | sensor_readings_24.csv | 100.00% | US1 to US24 | 99.36% | Not Saved |
| Gradient Boosting Classifier | GradientBoostingClassifier | sensor_readings_24.csv | 100.00% | US1 to US24 | 99.54% | ⭐⭐ **Best (Saved)** |
| Multi-Layer Perceptron (MLP) Classifier | MLPClassifier | sensor_readings_24.csv | 99.50% | US1 to US24 | 92.22% | Not Saved |