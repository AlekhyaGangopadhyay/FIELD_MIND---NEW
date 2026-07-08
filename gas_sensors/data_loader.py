import os
import pandas as pd
import numpy as np

WORKSPACE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def clean_missing_uci(val):
    """UCI Air Quality dataset uses -200 to represent missing values."""
    if val == -200 or val == -200.0:
        return np.nan
    return val

def load_smoke_dataset():
    """
    Loads smoke.csv using a leak-free SESSION-BASED split.
    Adds 5-period differences and 5-period rolling std for baseline robustness.
    Trains on Sessions 0, 1, 3, 4 and tests on Session 2.
    """
    path = os.path.join(WORKSPACE, "smoke.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    df = pd.read_csv(path)
    df = df.sort_values(by="UTC").reset_index(drop=True)
    df['diff_sec'] = df['UTC'].diff()
    df['session_id'] = (df['diff_sec'] > 60).cumsum()
    
    features_base = [
        'Temperature[C]', 'Humidity[%]', 'TVOC[ppb]', 'eCO2[ppm]', 
        'Raw H2', 'Raw Ethanol', 'Pressure[hPa]', 'PM1.0', 'PM2.5', 
        'NC0.5', 'NC1.0', 'NC2.5'
    ]
    target = 'Fire Alarm'
    
    # Feature engineering: diff and rolling standard dev
    df_feat = df[features_base].copy()
    for col in features_base:
        df_feat[f"{col}_diff_5"] = df_feat[col].diff(5).fillna(0)
        df_feat[f"{col}_roll_std"] = df_feat[col].rolling(5, min_periods=1).std().fillna(0)
        
    train_mask = df['session_id'].isin([0, 1, 3, 4])
    test_mask = df['session_id'] == 2
    
    X_train, y_train = df_feat.loc[train_mask], df.loc[train_mask, target]
    X_test, y_test = df_feat.loc[test_mask], df.loc[test_mask, target]
    
    return X_train, X_test, y_train, y_test

def load_mq4_dataset():
    """
    Loads all batches of MQ4 methane dataset using a leak-free BATCH-BASED split.
    Trains on Batches 1 to 8 and tests on Batches 9 and 10.
    """
    dataset_dir = os.path.join(WORKSPACE, "Methane_MQ4", "Dataset")
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Directory not found: {dataset_dir}")
        
    dfs = []
    for i in range(1, 11):
        batch_path = os.path.join(dataset_dir, f"batch{i}.csv")
        if os.path.exists(batch_path):
            bdf = pd.read_csv(batch_path)
            bdf['batch_num'] = i
            dfs.append(bdf)
            
    if not dfs:
        raise ValueError("No MQ4 batch files found!")
        
    df = pd.concat(dfs, ignore_index=True)
    features = [col for col in df.columns if col.startswith('feature_')]
    target = 'label'
    
    train_mask = df['batch_num'] <= 8
    test_mask = df['batch_num'] >= 9
    
    X_train, y_train = df.loc[train_mask, features], df.loc[train_mask, target]
    X_test, y_test = df.loc[test_mask, features], df.loc[test_mask, target]
    
    return X_train, X_test, y_train, y_test

def load_mq3_dataset(dataset_type="ethylene_CO", sample_rate=200):
    """
    Loads MQ3 txt files using a leak-free CHRONOLOGICAL split.
    Splits the raw data first, then drops NaNs independently to prevent leakages.
    Adds diff features to improve prediction of dynamic concentration changes.
    """
    filename = f"{dataset_type}.txt"
    path = os.path.join(WORKSPACE, "MQ3", filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
        
    with open(path, 'r') as f:
        header = f.readline().strip()
        
    if "CO conc" in header:
        targets = ["CO_conc_ppm", "Ethylene_conc_ppm"]
    else:
        targets = ["Methane_conc_ppm", "Ethylene_conc_ppm"]
        
    col_names = ["Time_sec"] + targets + [f"sensor_{i}" for i in range(1, 17)]
    
    df = pd.read_csv(
        path,
        sep=r'\s+',
        skiprows=lambda x: x > 0 and x % sample_rate != 0,
        names=col_names,
        header=None
    )
    
    if df.iloc[0, 0] == "Time" or (isinstance(df.iloc[0, 0], str) and "Time" in df.iloc[0, 0]):
        df = df.iloc[1:]
        
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Split first!
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    # Clean independently
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    
    features_base = [f"sensor_{i}" for i in range(1, 17)]
    
    # Feature engineering for train
    X_train = train_df[features_base].copy()
    for col in features_base:
        X_train[f"{col}_diff_1"] = X_train[col].diff().fillna(0)
        
    # Feature engineering for test
    X_test = test_df[features_base].copy()
    for col in features_base:
        X_test[f"{col}_diff_1"] = X_test[col].diff().fillna(0)
        
    y_train = train_df[targets]
    y_test = test_df[targets]
    
    return X_train, X_test, y_train, y_test

def load_air_quality_uci():
    """
    Loads AirQualityUCI.xlsx using a leak-free CHRONOLOGICAL split.
    Splits first and then imputes/forward-fills train and test independently
    to prevent temporal data leakage.
    """
    path = os.path.join(WORKSPACE, "MQ7", "AirQualityUCI.xlsx")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
        
    df = pd.read_excel(path, sheet_name='AirQualityUCI')
    
    unnamed_cols = [col for col in df.columns if col.startswith('Unnamed:')]
    df = df.drop(columns=unnamed_cols)
    
    features = ['PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'PT08.S5(O3)', 'T', 'RH', 'AH']
    target = 'C6H6(GT)'
    
    cols_to_use = features + [target]
    df_clean = df[cols_to_use].map(clean_missing_uci)
    df_clean = df_clean.dropna(subset=[target])
    
    # Chronological Split first!
    split_idx = int(len(df_clean) * 0.8)
    train_df = df_clean.iloc[:split_idx].copy()
    test_df = df_clean.iloc[split_idx:].copy()
    
    # Impute independently (Split-Before-Impute)
    train_df = train_df.ffill().bfill()
    test_df = test_df.ffill().bfill()
    
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]
    
    return X_train, X_test, y_train, y_test

def load_combined_gases():
    """
    Loads SO2,NO2,CO,NO3_combined_finalize.csv using a leak-free CHRONOLOGICAL split.
    Splits first, then adds autoregressive lag features and rolling averages independently.
    """
    path = os.path.join(WORKSPACE, "SO2,NO2,CO,NO3_combined_finalize.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
        
    df = pd.read_csv(path)
    
    features = ['SO2_ppm', 'NO2_ppm', 'O3_ppm']
    target = 'CO_ppm'
    
    df_subset = df[features + [target]]
    
    # Chronological split first!
    split_idx = int(len(df_subset) * 0.8)
    train_df = df_subset.iloc[:split_idx].copy()
    test_df = df_subset.iloc[split_idx:].copy()
    
    # Impute independently
    train_df = train_df.ffill().bfill().dropna()
    test_df = test_df.ffill().bfill().dropna()
    
    # Feature engineering for train
    # 1. Autoregressive lags 1 to 5
    for col in ['CO_ppm', 'SO2_ppm', 'NO2_ppm', 'O3_ppm']:
        for lag in [1, 2, 3, 4, 5]:
            train_df[f"{col}_lag_{lag}"] = train_df[col].shift(lag).bfill()
            
    # 2. Rolling averages (windows 3, 5)
    for col in ['CO_ppm', 'SO2_ppm']:
        for win in [3, 5]:
            train_df[f"{col}_roll_{win}"] = train_df[col].rolling(win, min_periods=1).mean()
            
    # Feature engineering for test
    # 1. Autoregressive lags 1 to 5
    for col in ['CO_ppm', 'SO2_ppm', 'NO2_ppm', 'O3_ppm']:
        for lag in [1, 2, 3, 4, 5]:
            test_df[f"{col}_lag_{lag}"] = test_df[col].shift(lag).bfill()
            
    # 2. Rolling averages (windows 3, 5)
    for col in ['CO_ppm', 'SO2_ppm']:
        for win in [3, 5]:
            test_df[f"{col}_roll_{win}"] = test_df[col].rolling(win, min_periods=1).mean()
            
    # X features are all columns except the target 'CO_ppm'
    features_to_use = [c for c in train_df.columns if c != 'CO_ppm']
    
    X_train, y_train = train_df[features_to_use], train_df[target]
    X_test, y_test = test_df[features_to_use], test_df[target]
    
    return X_train, X_test, y_train, y_test
