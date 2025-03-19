import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import os

# Configuration
ku_har_file = "D:\\KU-HAR\\Time_domain_subsamples\\KU-HAR_time_domain_subsamples_20750x300.csv"
output_file = "ku_har_accel_ABCDE_features.csv"
sampling_rate = 100  # KU-HAR sampling rate (100 Hz)

print(f"Loading KU-HAR dataset from {ku_har_file}")

# Load the KU-HAR dataset
try:
    # Based on the output, we can see the dataset has 1803 columns
    # We need to load the data with proper column names
    column_names = [f"acc_x_{i}" for i in range(300)]
    column_names += [f"acc_y_{i}" for i in range(300)]
    column_names += [f"acc_z_{i}" for i in range(300)]
    column_names += [f"gyro_x_{i}" for i in range(300)]
    column_names += [f"gyro_y_{i}" for i in range(300)]
    column_names += [f"gyro_z_{i}" for i in range(300)]
    column_names += ["class_id", "length", "serial_no"]
    
    # Load the data with the correct column names
    ku_har_df = pd.read_csv(ku_har_file, header=None, names=column_names)
    print(f"Successfully loaded {len(ku_har_df)} samples from KU-HAR dataset")
    print(f"Dataset shape: {ku_har_df.shape}")
    
    # Show the distribution of class IDs in the dataset
    class_id_counts = ku_har_df["class_id"].value_counts().sort_index()
    print("\nClass ID distribution in the dataset:")
    for class_id, count in class_id_counts.items():
        print(f"  Class ID {class_id}: {count} samples")
    
except Exception as e:
    print(f"Error loading KU-HAR dataset: {e}")
    exit(1)

# Activity mapping from KU-HAR to WISDM categories
# Based on the description provided
activity_mapping = {
    11: 'A',  # Walk
    12: 'A',  # Walk-backward
    13: 'A',  # Walk-circle
    14: 'B',  # Run
    15: 'C',  # Stair-up
    16: 'C',  # Stair-down
    4: 'D',   # Stand-sit (treated as both D and E)
}

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
        
        # Jerk as difference of acceleration
        jerk = np.diff(window[axis].values, prepend=window[axis].values[0])
        features[f"accel_{axis}_JERK_AVG"] = np.mean(np.abs(jerk))
        features[f"accel_{axis}_JERK_VAR"] = np.var(jerk)
    
    # |a|
    features["accel_RESULTANT"] = np.sqrt(window['x']**2 + window['y']**2 + window['z']**2).mean()
    
    # Correlation
    features["accel_XY_CORR"] = window['x'].corr(window['y'])
    features["accel_XZ_CORR"] = window['x'].corr(window['z'])
    features["accel_YZ_CORR"] = window['y'].corr(window['z'])
    
    # Fast Fourier Transform
    for axis in ['x', 'y', 'z']:
        fft_values = np.abs(np.fft.fft(window[axis].values))**2
        features[f"accel_{axis}_ENERGY"] = np.sum(fft_values) / len(fft_values)
    
    return features

print("Processing KU-HAR data...")
all_features = []
processed_count = 0
skipped_count = 0

# Process each row in the dataset
for idx, row in ku_har_df.iterrows():
    # Get the class ID
    class_id = int(row["class_id"])
    
    # Skip samples with activities not in our mapping
    if class_id not in activity_mapping:
        skipped_count += 1
        if skipped_count <= 5:
            print(f"Skipping sample with class ID {class_id} (not in mapping)")
        continue
    
    # Map the activity to our A-E categories
    activity = activity_mapping[class_id]
    
    # Extract accelerometer data
    x_data = np.array([row[f"acc_x_{i}"] for i in range(300)])
    y_data = np.array([row[f"acc_y_{i}"] for i in range(300)])
    z_data = np.array([row[f"acc_z_{i}"] for i in range(300)])
    
    # Create a dataframe for this sample's data
    accel_df = pd.DataFrame({
        'x': x_data,
        'y': y_data,
        'z': z_data
    })
    
    # Compute features
    features = compute_accel_features(accel_df)
    
    # Add activity and subject information
    features['activity'] = activity
    features['subject'] = f"ku_har_{idx}"  # Create a unique subject ID
    
    # For class_id 4 (Stand-sit), create another sample with activity 'E'
    if class_id == 4:
        features_e = features.copy()
        features_e['activity'] = 'E'
        all_features.append(features_e)
    
    all_features.append(features)
    processed_count += 1
    
    if processed_count % 500 == 0:
        print(f"Processed {processed_count} samples")

# Create a DataFrame with all features
if len(all_features) > 0:
    feature_df = pd.DataFrame(all_features)
    
    # Display distribution of activities
    activity_counts = feature_df['activity'].value_counts()
    print("\nFinal activity distribution in feature set:")
    for activity, count in activity_counts.items():
        print(f"  {activity}: {count} samples")
    
    # Save features to CSV
    feature_df.to_csv(output_file, index=False)
    print(f"Features saved to {output_file}")
    print(f"Feature set contains {len(feature_df)} samples with {len(feature_df.columns)} features")
    
    # Save activity features (without subject)
    activity_features = feature_df.drop('subject', axis=1)
    activity_output_file = "ku_har_activity_accel_ABCDE_features.csv"
    activity_features.to_csv(activity_output_file, index=False)
    print(f"Activity features saved to {activity_output_file}")
else:
    print("No features were successfully extracted.")
    print(f"Total samples processed: {processed_count}")
    print(f"Total samples skipped: {skipped_count}")

print("Done")