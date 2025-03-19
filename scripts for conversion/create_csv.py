import os
import pandas as pd
import numpy as np
from scipy import stats

# Directory containing the dataset
data_dir = r"C:\Users\gontu\OneDrive\Documents\WISDM from site\wisdm-dataset\wisdm-dataset\raw\phone\accel"

# Target activities to filter
target_activities = ['A', 'B', 'C', 'D', 'E']

all_data = []
for filename in os.listdir(data_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(data_dir, filename)
        try:
            # Read the file line by line
            rows = []
            with open(file_path, 'r') as f:
                for line in f:
                    # Strip whitespace and remove trailing semicolon if present
                    line = line.strip()
                    if line.endswith(';'):
                        line = line[:-1]
                    
                    # Skip empty lines
                    if not line:
                        continue
                    
                    # Split by comma
                    parts = line.split(',')
                    
                    # Ensure we have exactly 6 parts
                    if len(parts) == 6:
                        rows.append(parts)
            
            if rows:
                # Create DataFrame
                df = pd.DataFrame(rows, columns=['subject', 'activity', 'timestamp', 'x', 'y', 'z'])
                
                # Clean and convert data types
                df['subject'] = pd.to_numeric(df['subject'], errors='coerce')
                df['activity'] = df['activity'].astype(str)
                
                # Convert timestamp from nanoseconds to milliseconds
                df['timestamp'] = (pd.to_numeric(df['timestamp'], errors='coerce') / 1e6).astype('int64')
                
                # Convert acceleration values
                for col in ['x', 'y', 'z']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Drop rows with NaN values
                df = df.dropna()
                
                # Filter by activity
                df = df[df['activity'].isin(target_activities)]
                
                if not df.empty:
                    all_data.append(df)
                    print(f"Loaded {len(df)} records from {filename}")
                        
        except Exception as e:
            print(f"Error loading {filename}: {e}")

# Concatenate all data if any exists
if all_data:
    df = pd.concat(all_data)

    # Get the original count before processing
    original_count = len(df)
    print(f"Total records before processing: {original_count}")

    # Handle duplicate timestamps by averaging their values
    print("Handling duplicate timestamps...")
    df_grouped = df.groupby(['subject', 'activity', 'timestamp']).agg({
        'x': 'mean',
        'y': 'mean',
        'z': 'mean'
    }).reset_index()
    
    duplicates_removed = original_count - len(df_grouped)
    print(f"Removed {duplicates_removed} duplicate timestamp records")
    
    # Find min and max values for each column with the full record
    print("\n--- Records with Min/Max Values ---")
    
    # For timestamp
    min_timestamp = df_grouped['timestamp'].min()
    max_timestamp = df_grouped['timestamp'].max()
    min_timestamp_record = df_grouped[df_grouped['timestamp'] == min_timestamp]
    max_timestamp_record = df_grouped[df_grouped['timestamp'] == max_timestamp]
    
    print("\nRecord with Minimum timestamp:")
    print(min_timestamp_record)
    print("\nRecord with Maximum timestamp:")
    print(max_timestamp_record)
    
    # For x acceleration
    min_x = df_grouped['x'].min()
    max_x = df_grouped['x'].max()
    min_x_record = df_grouped[df_grouped['x'] == min_x]
    max_x_record = df_grouped[df_grouped['x'] == max_x]
    
    print("\nRecord with Minimum x acceleration:")
    print(min_x_record)
    print("\nRecord with Maximum x acceleration:")
    print(max_x_record)
    
    # For y acceleration
    min_y = df_grouped['y'].min()
    max_y = df_grouped['y'].max()
    min_y_record = df_grouped[df_grouped['y'] == min_y]
    max_y_record = df_grouped[df_grouped['y'] == max_y]
    
    print("\nRecord with Minimum y acceleration:")
    print(min_y_record)
    print("\nRecord with Maximum y acceleration:")
    print(max_y_record)
    
    # For z acceleration
    min_z = df_grouped['z'].min()
    max_z = df_grouped['z'].max()
    min_z_record = df_grouped[df_grouped['z'] == min_z]
    max_z_record = df_grouped[df_grouped['z'] == max_z]
    
    print("\nRecord with Minimum z acceleration:")
    print(min_z_record)
    print("\nRecord with Maximum z acceleration:")
    print(max_z_record)
    
    # Create a dictionary to store all min/max records
    extreme_records = {
        'min_timestamp': min_timestamp_record,
        'max_timestamp': max_timestamp_record,
        'min_x': min_x_record,
        'max_x': max_x_record,
        'min_y': min_y_record,
        'max_y': max_y_record,
        'min_z': min_z_record,
        'max_z': max_z_record
    }
    
    # Save all extreme records to separate CSV files
    os.makedirs('extreme_records', exist_ok=True)
    for name, record in extreme_records.items():
        record.to_csv(f"extreme_records/{name}.csv", index=False)
    
    print("\nExtreme records saved to extreme_records folder")
    
    # Also save a combined file with all extreme records
    combined_extremes = pd.concat(extreme_records.values())
    combined_extremes.to_csv("extreme_records/all_extreme_records.csv", index=False)
    print("Combined extreme records saved to extreme_records/all_extreme_records.csv")
    
    # Identify outliers using z-score method
    print("\nIdentifying outliers...")
    # Create a copy of the dataframe for outlier detection
    df_outliers = df_grouped.copy()
    
    # Function to detect outliers using z-score
    def detect_outliers(df, column, threshold=3):
        z_scores = np.abs(stats.zscore(df[column]))
        return df[z_scores > threshold]
    
    # Detect outliers in each acceleration column
    outliers_x = detect_outliers(df_outliers, 'x')
    outliers_y = detect_outliers(df_outliers, 'y')
    outliers_z = detect_outliers(df_outliers, 'z')
    
    # Combine all outliers
    all_outliers = pd.concat([outliers_x, outliers_y, outliers_z]).drop_duplicates()
    
    print(f"Detected {len(all_outliers)} potential outlier records")
    
    # Save outliers to a separate file
    if not all_outliers.empty:
        all_outliers.to_csv("outliers.csv", index=False)
        print("Outliers saved to outliers.csv")
    
    # Save the processed data
    df_grouped.to_csv("processed_data.csv", index=False)
    print(f"Saved {len(df_grouped)} processed records to processed_data.csv")
else:
    print("No valid data was loaded.")