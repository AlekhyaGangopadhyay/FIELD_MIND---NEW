# Model Performance & Metrics Table

| Model Name | Algorithm Used | Dataset Used | Features Used | Train Accuracy | Test/Evaluation Accuracy |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **Occupancy Classifier** | **Random Forest Classifier** | UCI Occupancy Detection (`datatraining.txt`, `datatest.txt`, `datatest2.txt`) | `Temperature`, `Humidity`, `Light`, `CO2`, `HumidityRatio`, plus 18 temporal rolling and interaction features. | **99.01%** | **97.11%** (UCI Test Set 1)<br>**96.80%** (UCI Test Set 2) |
| **IoT Anomaly Detector** | **Isolation Forest** (Unsupervised) | IoT Telemetry Dataset (`iot_telemetry_data.xlsx`) | `temp`, `humidity`, `temp_hum_product`, `temp_hum_ratio`, `humidex`, and rolling mean/std features. | **93.38%*** | **93.38%*** (IoT Dataset)<br>**67.56%*** (Raspberry PI Logs - Cross-Domain) |
| **UCI Anomaly Detector** | **Isolation Forest** (Unsupervised) | UCI Environmental Dataset (`datatraining.txt`, `datatest.txt`) | `Temperature`, `Humidity`, `CO2`, `temp_hum_product`, `temp_hum_ratio`, `humidex`, `temp_co2_product`, and rolling mean/std features. | **73.87%*** | **89.19%*** (UCI Test Set 1)<br>**69.94%*** (Raspberry PI Logs - Respective-Domain) |

*\*Note: For unsupervised Isolation Forest models, "Accuracy" represents the model's classification alignment rate against predefined domain safety guideline boundaries (e.g. Temperature, Humidity, and CO2 threshold alerts).*
