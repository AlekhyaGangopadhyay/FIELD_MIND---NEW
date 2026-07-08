import struct
import numpy as np
import pandas as pd
import os

def ibm_to_ieee(ibm_uint32):
    """
    Converts big-endian IBM float (represented as uint32) to IEEE float32.
    IBM float format:
      - 1 bit sign
      - 7 bits base-16 exponent
      - 24 bits fraction
    Formula: (-1)^sign * fraction * 16^(exponent - 70)
    """
    sign = (ibm_uint32 >> 31) & 1
    exponent = (ibm_uint32 >> 24) & 0x7f
    fraction = ibm_uint32 & 0x00ffffff
    
    # Avoid powers of 16 that underflow or overflow by doing standard math on float32
    val = fraction.astype(np.float32) * (16.0 ** (exponent.astype(np.float32) - 70.0))
    val = np.where(sign == 1, -val, val)
    return val

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sgy_path = os.path.join(script_dir, "data", "blasts.sgy")
    blasts_txt_path = os.path.join(script_dir, "data", "BLASTS.txt")
    output_csv = os.path.join(script_dir, "data", "vibration_features.csv")
    
    print("Step 1: Reading and aggregating BLASTS.txt...")
    df_blasts = pd.read_csv(blasts_txt_path, sep='\t')
    
    # Group by blast to extract blast-level parameters
    # detonator: 'E' for Electronic, 'N' for Non-electric. Let's map to binary: E=1, N=0.
    df_blasts['detonator_code'] = df_blasts['detonator'].map({'E': 1, 'N': 0}).fillna(0).astype(int)
    
    blast_groups = df_blasts.groupby('blast')
    df_blast_features = pd.DataFrame({
        'total_charge': blast_groups['charge'].sum(),
        'max_charge': blast_groups['charge'].max(),
        'num_holes': blast_groups.size(),
        'detonator_code': blast_groups['detonator_code'].first()
    }).reset_index()
    
    print(f"Aggregated {len(df_blast_features)} unique blasts from BLASTS.txt")
    
    print("Step 2: Processing blasts.sgy traces...")
    traces = []
    num_traces = 14451
    trace_size = 10240 # 240-byte header + 2500 samples * 4 bytes
    
    with open(sgy_path, "rb") as f:
        f.seek(3600) # Skip 3600-byte EBCDIC + binary file headers
        
        for i in range(num_traces):
            hdr_data = f.read(240)
            if not hdr_data:
                print(f"Warning: Reached EOF early at trace {i}")
                break
            
            # Unpack header fields
            fldr = struct.unpack(">i", hdr_data[8:12])[0]       # blast sequence number
            tracf = struct.unpack(">i", hdr_data[12:16])[0]     # receiver station number
            trid = struct.unpack(">h", hdr_data[28:30])[0]      # trace ID: 12=Z, 13=EW, 14=NS
            offset = struct.unpack(">i", hdr_data[36:40])[0]    # source-receiver offset
            gelev = struct.unpack(">i", hdr_data[40:44])[0]     # group elevation
            selev = struct.unpack(">i", hdr_data[44:48])[0]     # source elevation
            c_scalar = struct.unpack(">h", hdr_data[70:72])[0]  # coordinate scalar
            sx = struct.unpack(">i", hdr_data[72:76])[0]        # source coordinate X
            sy = struct.unpack(">i", hdr_data[76:80])[0]        # source coordinate Y
            gx = struct.unpack(">i", hdr_data[80:84])[0]        # group coordinate X
            gy = struct.unpack(">i", hdr_data[84:88])[0]        # group coordinate Y
            
            # Apply coordinate scalar
            scale = 1.0
            if c_scalar < 0:
                scale = 1.0 / abs(c_scalar)
            elif c_scalar > 0:
                scale = float(c_scalar)
                
            sx_s = sx * scale
            sy_s = sy * scale
            gx_s = gx * scale
            gy_s = gy * scale
            gelev_s = gelev * scale
            selev_s = selev * scale
            
            # Read trace data (2500 samples of 4-byte IBM float)
            trace_raw = f.read(10000)
            ibm_data = np.frombuffer(trace_raw, dtype='>u4')
            ieee_data = ibm_to_ieee(ibm_data)
            
            # Target metric: Peak Particle Velocity (PPV)
            ppv = np.max(np.abs(ieee_data))
            
            traces.append({
                'blast': fldr,
                'tracf': tracf,
                'trid': trid,
                'offset': offset,
                'gelev': gelev_s,
                'selev': selev_s,
                'sx': sx_s,
                'sy': sy_s,
                'gx': gx_s,
                'gy': gy_s,
                'ppv': ppv
            })
            
            if (i + 1) % 3000 == 0:
                print(f"  Processed {i+1} / {num_traces} traces...")
                
    df_traces = pd.DataFrame(traces)
    print(f"Processed {len(df_traces)} traces from SEG-Y file.")
    
    print("Step 3: Merging datasets and engineering features...")
    # Merge trace information with blast parameters
    df_merged = pd.merge(df_traces, df_blast_features, on='blast', how='left')
    
    # Check for any missing values from the merge
    if df_merged.isnull().any().any():
        print("Warning: Missing values detected after merge! Filling with default values.")
        df_merged['total_charge'] = df_merged['total_charge'].fillna(df_merged['total_charge'].median())
        df_merged['max_charge'] = df_merged['max_charge'].fillna(df_merged['max_charge'].median())
        df_merged['num_holes'] = df_merged['num_holes'].fillna(df_merged['num_holes'].median())
        df_merged['detonator_code'] = df_merged['detonator_code'].fillna(0).astype(int)
    
    # Feature Engineering
    # 1. USBM Scaled Distance: SD_USBM = Distance / sqrt(Max_Charge)
    # 2. Langefors-Kihlstrom Scaled Distance: SD_LK = Distance / (Max_Charge ^ (1/3))
    # Make sure we don't divide by zero
    max_charge_eps = np.maximum(df_merged['max_charge'].values, 1e-5)
    df_merged['scaled_distance_usbm'] = df_merged['offset'] / np.sqrt(max_charge_eps)
    df_merged['scaled_distance_langefors'] = df_merged['offset'] / (max_charge_eps ** (1.0/3.0))
    
    # 3. Geometric height differences
    df_merged['elevation_diff'] = df_merged['gelev'] - df_merged['selev']
    
    # 4. Target variable for classification: ppv > 1.0 (vibration hazard)
    df_merged['vibration_hazard'] = (df_merged['ppv'] > 1.0).astype(int)
    
    # Save to disk
    df_merged.to_csv(output_csv, index=False)
    print(f"Step 4: Preprocessing complete. Unified dataset saved to {output_csv}")
    print("Dataset stats:")
    print(f"  Total records: {len(df_merged)}")
    print(f"  Classification target (vibration_hazard > 1.0 mm/s) ratio: {df_merged['vibration_hazard'].mean():.4f}")
    print(f"  PPV Min: {df_merged['ppv'].min():.4f}, Max: {df_merged['ppv'].max():.4f}, Mean: {df_merged['ppv'].mean():.4f}")

if __name__ == "__main__":
    main()
