import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from collections import Counter, deque

accel_dir = r"C:\Users\gontu\OneDrive\Documents\WISDM from site\wisdm-dataset\wisdm-dataset\raw\phone\accel"


sampling_rate = 20  
window_size_seconds = 5 
window_size = sampling_rate * window_size_seconds  
activity_threshold = 0.85 


target_activities = ['A', 'B', 'C', 'D', 'E']

def load_data(directory):
    all_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            try:
                
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                
                rows = []
                for line in lines:
                    # Remove semicolon
                    if line.strip().endswith(';'):
                        line = line.strip()[:-1]
                    
                    # Split at ,
                    parts = line.strip().split(',')
                    if len(parts) == 6:  
                        rows.append(parts)
                
                if rows:
                    # Create into a DataFrame
                    df = pd.DataFrame(rows, columns=['subject', 'activity', 'timestamp', 'x', 'y', 'z'])
                    
                    
                    df['subject'] = df['subject'].astype(str)
                    df['activity'] = df['activity'].astype(str)
                    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
                    df['x'] = pd.to_numeric(df['x'], errors='coerce')
                    df['y'] = pd.to_numeric(df['y'], errors='coerce')
                    df['z'] = pd.to_numeric(df['z'], errors='coerce')
                    
                    # drop rows with missing values
                    df = df.dropna()
                    
                    # Filter
                    df = df[df['activity'].isin(target_activities)]
                    
                    if not df.empty:
                        all_data.append(df)
                        print(f"Loaded {len(df)} records from {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame(columns=['subject', 'activity', 'timestamp', 'x', 'y', 'z'])

print("Loading accelerometer data for activities", target_activities)
accel_df = load_data(accel_dir)
print(f"Total accelerometer data: {len(accel_df)} records")


activity_counts = accel_df['activity'].value_counts()
print("Activity distribution:")
for activity, count in activity_counts.items():
    print(f"  {activity}: {count} samples")



def calculate_peak_intervals(signal):
    """Calculate the average time between peaks in milliseconds"""
    # Find peaks
    peaks, _ = find_peaks(signal, distance=3)
    
    if len(peaks) < 2:
        return 0  # Not enough peaks
    
    # Calculate intervals 
    intervals = np.diff(peaks) * (1000 / sampling_rate)
    return np.mean(intervals)

def compute_accel_features(window):
    """Compute features for accelerometer data"""
    features = {}
    
    # Basic features for each axis
    for axis in ['x', 'y', 'z']:
        features[f"accel_{axis}_AVG"] = window[axis].mean()
        features[f"accel_{axis}_STANDDEV"] = window[axis].std()
        features[f"accel_{axis}_VAR"] = window[axis].var()
        features[f"accel_{axis}_ABSOLDEV"] = np.mean(np.abs(window[axis] - window[axis].mean()))
        features[f"accel_{axis}_PEAK"] = calculate_peak_intervals(window[axis].values)
        
        #  jerk as difference of acceleration
        jerk = np.diff(window[axis].values, prepend=window[axis].values[0])
        features[f"accel_{axis}_JERK_AVG"] = np.mean(np.abs(jerk))
        features[f"accel_{axis}_JERK_VAR"] = np.var(jerk)
    
    # |a|
    features["accel_RESULTANT"] = np.sqrt(window['x']**2 + window['y']**2 + window['z']**2).mean()
    
    # Correlation
    features["accel_XY_CORR"] = window['x'].corr(window['y'])
    features["accel_XZ_CORR"] = window['x'].corr(window['z'])
    features["accel_YZ_CORR"] = window['y'].corr(window['z'])
    
    # fast fourier transform
    for axis in ['x', 'y', 'z']:
        fft_values = np.abs(np.fft.fft(window[axis].values))**2
        features[f"accel_{axis}_ENERGY"] = np.sum(fft_values) / len(fft_values)
    
    return features


print("Processing data efficiently...")
all_features = []

# by subject
subjects = accel_df['subject'].unique()
print(f"Processing {len(subjects)} subjects")

for subject in subjects:
    print(f"Processing subject {subject}...")
    
    
    subject_accel = accel_df[accel_df['subject'] == subject].copy()
    
    
    if len(subject_accel) < window_size:
        print(f"  Not enough accelerometer data for subject {subject}")
        continue
    
    
    subject_accel.sort_values('timestamp', inplace=True)
    
    
    for activity, activity_accel in subject_accel.groupby('activity'):
        
        if activity not in target_activities:
            continue
            
        
        if len(activity_accel) < window_size:
            continue
        
        print(f"  Processing activity {activity} with {len(activity_accel)} data points")
        activity_accel = activity_accel.sort_values('timestamp')
        
        
        step_size = window_size // 2
        
        for i in range(0, len(activity_accel) - window_size + 1, step_size):
            accel_window = activity_accel.iloc[i:i + window_size]
            
            
            activity_counts = Counter(accel_window['activity'])
            most_common, count = activity_counts.most_common(1)[0]
            consistency = count / len(accel_window)
            
            if consistency >= activity_threshold:
                # Compute features
                accel_feats = compute_accel_features(accel_window)
                
                
                accel_feats['subject'] = subject
                accel_feats['activity'] = most_common
                
                all_features.append(accel_feats)


feature_df = pd.DataFrame(all_features)


activity_counts = feature_df['activity'].value_counts()
print("\nFinal activity distribution in feature set:")
for activity, count in activity_counts.items():
    print(f"  {activity}: {count} samples")

output_file = "wisdm_accel_ABCDE_features.csv"
feature_df.to_csv(output_file, index=False)
print(f"Features saved to {output_file}")
print(f"Feature set contains {len(feature_df)} samples with {len(feature_df.columns)} features")

# saving activity features
activity_features = feature_df.drop('subject', axis=1)
activity_output_file = "wisdm_activity_accel_ABCDE_features.csv"
activity_features.to_csv(activity_output_file, index=False)
print(f"Activity features saved to {activity_output_file}")

print("Done")
