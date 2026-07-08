import pandas as pd
import numpy as np
import os
import re

def compute_rolling_features(df, cols, windows=[5, 30]):
    """
    Computes rolling mean and standard deviation for specified columns and window sizes.
    """
    for col in cols:
        if col in df.columns:
            for w in windows:
                df[f'{col}_roll_mean_{w}'] = df[col].rolling(window=w, min_periods=1).mean()
                df[f'{col}_roll_std_{w}'] = df[col].rolling(window=w, min_periods=1).std().fillna(0.0)
    return df

def add_interaction_features(df, temp_col, hum_col):
    """
    Adds environmental interaction features: product and ratio.
    """
    if temp_col in df.columns and hum_col in df.columns:
        # Avoid division by zero
        hum_eps = df[hum_col] + 1e-5
        df['temp_hum_product'] = df[temp_col] * df[hum_col]
        df['temp_hum_ratio'] = df[temp_col] / hum_eps
        # Simple Humidex approximation: Heat index based on temp and relative humidity
        # Humidex = Temp + 5/9 * (e - 10), where e is vapor pressure (approximate formula below)
        # e = 6.11 * exp(5417.7530 * (1/273.16 - 1/(273.15 + Temp))) * (Humidity / 100)
        try:
            e = 6.11 * np.exp(5417.7530 * (1/273.16 - 1/(273.15 + df[temp_col]))) * (df[hum_col] / 100.0)
            df['humidex'] = df[temp_col] + (5.0 / 9.0) * (e - 10.0)
        except Exception:
            # Fallback to a simpler combined safety index if math fails
            df['humidex'] = df[temp_col] + 0.55 * (df[hum_col] - 50.0)
    return df

def preprocess_iot_data(input_path, output_path):
    """
    Cleans and preprocesses the Kaggle IoT Telemetry dataset.
    """
    print(f"Preprocessing IoT Telemetry data from {input_path}...")
    # This is a large file, read Excel using openpyxl
    df = pd.read_excel(input_path)
    
    # Sort by timestamp to ensure correct temporal order
    df = df.sort_values(by='ts').reset_index(drop=True)
    
    # Parse timestamp
    df['datetime'] = pd.to_datetime(df['ts'], unit='s')
    
    # Compute interaction features
    df = add_interaction_features(df, 'temp', 'humidity')
    
    # Compute rolling features
    df = compute_rolling_features(df, ['temp', 'humidity'])
    
    # Save cleaned data to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"IoT Telemetry preprocessed successfully. Shape: {df.shape}. Saved to {output_path}")
    return df

def preprocess_uci_data(train_path, test_path, test2_path, output_dir):
    """
    Cleans and preprocesses the UCI Occupancy Detection dataset files.
    """
    print(f"Preprocessing UCI Occupancy datasets from {os.path.dirname(train_path)}...")
    os.makedirs(output_dir, exist_ok=True)
    
    processed_paths = {}
    for name, path in [('train', train_path), ('test', test_path), ('test2', test2_path)]:
        df = pd.read_csv(path)
        
        # Sort by date
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date').reset_index(drop=True)
        
        # Time components
        df['hour'] = df['date'].dt.hour
        df['dayofweek'] = df['date'].dt.dayofweek
        
        # Interaction features
        df = add_interaction_features(df, 'Temperature', 'Humidity')
        if 'CO2' in df.columns:
            df['temp_co2_product'] = df['Temperature'] * df['CO2']
            
        # Rolling features
        df = compute_rolling_features(df, ['Temperature', 'Humidity', 'CO2'])
        
        out_path = os.path.join(output_dir, f"uci_{name}_clean.csv")
        df.to_csv(out_path, index=False)
        processed_paths[name] = out_path
        print(f"UCI {name} preprocessed. Shape: {df.shape}. Saved to {out_path}")
        
    return processed_paths

def preprocess_raspberry_pi_data(input_path, output_path):
    """
    Cleans and preprocesses the Raspberry PI logs.
    """
    print(f"Preprocessing Raspberry PI logs from {input_path}...")
    
    # Read raw data without header
    df = pd.read_csv(input_path, header=None, names=['Date', 'Time', 'Temp_Raw', 'Hum_Raw'])
    
    # Parse temperature and humidity from string formats (e.g. 'T=22.0', 'H=20.0')
    df['temp'] = df['Temp_Raw'].str.extract(r'T=([0-9\.\-]+)').astype(float)
    df['humidity'] = df['Hum_Raw'].str.extract(r'H=([0-9\.\-]+)').astype(float)
    
    # Handle missing values (e.g., using linear interpolation or forward/backward fill)
    df['temp'] = df['temp'].interpolate(method='linear').ffill().bfill()
    df['humidity'] = df['humidity'].interpolate(method='linear').ffill().bfill()
    
    # Parse DateTime
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%m/%d/%y %H:%M:%S', errors='coerce')
    # If standard parsing fails, fallback
    if df['datetime'].isnull().any():
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
    
    df = df.sort_values(by='datetime').reset_index(drop=True)
    
    # Compute interaction and rolling features
    df = add_interaction_features(df, 'temp', 'humidity')
    df = compute_rolling_features(df, ['temp', 'humidity'])
    
    # Drop raw string columns as they are no longer needed
    df = df.drop(columns=['Temp_Raw', 'Hum_Raw'])
    
    # Save cleaned data to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Raspberry PI logs preprocessed. Shape: {df.shape}. Saved to {output_path}")
    return df

if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    output_dir = os.path.join(base_dir, "data_clean")
    
    # Preprocess Kaggle IoT Telemetry Data
    iot_input = os.path.join(base_dir, "1 - Kaggle (Env)", "iot_telemetry_data.xlsx")
    iot_output = os.path.join(output_dir, "iot_telemetry_clean.csv")
    if os.path.exists(iot_input):
        preprocess_iot_data(iot_input, iot_output)
        
    # Preprocess UCI Occupancy Detection Data
    uci_train = os.path.join(base_dir, "2 - UCI", "datatraining.txt")
    uci_test = os.path.join(base_dir, "2 - UCI", "datatest.txt")
    uci_test2 = os.path.join(base_dir, "2 - UCI", "datatest2.txt")
    if os.path.exists(uci_train):
        preprocess_uci_data(uci_train, uci_test, uci_test2, output_dir)
        
    # Preprocess Raspberry PI logs
    rpi_input = os.path.join(base_dir, "4 - Kaggle (Raspberry PI)", "log_temp.csv")
    rpi_output = os.path.join(output_dir, "log_temp_clean.csv")
    if os.path.exists(rpi_input):
        preprocess_raspberry_pi_data(rpi_input, rpi_output)
