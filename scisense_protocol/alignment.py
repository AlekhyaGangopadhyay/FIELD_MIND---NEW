import pandas as pd
import numpy as np

class TemporalAligner:
    """
    Temporal aligner for synchronizing heterogeneous sensor streams 
    with different sampling frequencies (e.g. gas, vibration, environment, ultrasonic).
    """
    def __init__(self, time_window_seconds=1.0):
        self.time_window_seconds = time_window_seconds
        
    def align_streams(self, streams, start_time, end_time):
        """
        Aligns multiple sensor streams into synchronized intervals.
        
        Parameters:
        - streams: dict of (modality_name: list of dicts or DataFrame)
                   Each entry must contain 'timestamp' (datetime or epoch seconds) and features.
        - start_time: epoch seconds start
        - end_time: epoch seconds end
        
        Returns:
        - List of dicts representing aligned multi-modal epochs
        """
        # Create a range of target timestamps
        target_timestamps = np.arange(start_time, end_time, self.time_window_seconds)
        aligned_epochs = []
        
        # Convert all streams to pandas DataFrames with epoch index
        processed_streams = {}
        for name, data in streams.items():
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
                
            if df.empty:
                processed_streams[name] = pd.DataFrame()
                continue
                
            # Ensure timestamp is a numeric epoch float
            if 'timestamp' in df.columns:
                df['timestamp'] = df['timestamp'].astype(float)
                df = df.sort_values('timestamp')
                processed_streams[name] = df
            else:
                processed_streams[name] = pd.DataFrame()
                
        # Align each target timestamp window
        for t in target_timestamps:
            window_start = t
            window_end = t + self.time_window_seconds
            
            epoch_data = {
                'timestamp_start': window_start,
                'timestamp_end': window_end,
                'aligned_features': {}
            }
            
            for name, df in processed_streams.items():
                if df.empty:
                    epoch_data['aligned_features'][name] = None
                    continue
                    
                # Find observations in this window
                mask = (df['timestamp'] >= window_start) & (df['timestamp'] < window_end)
                window_obs = df.loc[mask]
                
                if not window_obs.empty:
                    # If multiple observations are in the window, average them
                    feature_cols = [c for c in df.columns if c != 'timestamp']
                    mean_features = window_obs[feature_cols].mean().to_dict()
                    epoch_data['aligned_features'][name] = mean_features
                else:
                    # Forward-fill: find the last observation before this window
                    prior_obs = df.loc[df['timestamp'] < window_start]
                    if not prior_obs.empty:
                        last_obs = prior_obs.iloc[-1]
                        feature_cols = [c for c in df.columns if c != 'timestamp']
                        epoch_data['aligned_features'][name] = last_obs[feature_cols].to_dict()
                    else:
                        epoch_data['aligned_features'][name] = None
                        
            aligned_epochs.append(epoch_data)
            
        return aligned_epochs
