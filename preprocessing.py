import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from scipy import stats
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Updated paths for your local system
accel_path = r"C:\Users\gontu\OneDrive\Documents\WISDM from site\wisdm-dataset\wisdm-dataset\raw\phone\accel"
gyro_path = r"C:\Users\gontu\OneDrive\Documents\WISDM from site\wisdm-dataset\wisdm-dataset\raw\phone\gyro"

def load_wisdm_data(data_path, sensor_type):
    """
    Load WISDM dataset with improved error handling and logging

    Parameters:
    data_path (str): Path to the sensor data files
    sensor_type (str): Type of sensor ('accel' or 'gyro')

    Returns:
    pd.DataFrame: Loaded sensor data
    """
    data = []
    error_count = 0
    file_count = 0

    logging.info(f"Loading {sensor_type} data from {data_path}")

    for filename in os.listdir(data_path):
        if not filename.endswith('.txt'):
            continue

        file_count += 1
        lines_processed = 0

        with open(os.path.join(data_path, filename)) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip().rstrip(';')
                parts = line.split(',')

                if len(parts) == 6:
                    try:
                        subject = int(parts[0])
                        activity = parts[1]
                        timestamp = int(parts[2])
                        x, y, z = map(float, parts[3:])
                        data.append([subject, activity, timestamp, x, y, z])
                        lines_processed += 1
                    except Exception as e:
                        error_count += 1
                        if error_count < 10:  # Limit logging to avoid flooding
                            logging.warning(f"Error in file {filename}, line {line_num}: {e}")
                else:
                    error_count += 1

        logging.info(f"Processed {filename}: {lines_processed} valid lines")

    logging.info(f"Total files processed: {file_count}")
    logging.info(f"Total errors encountered: {error_count}")

    if not data:
        logging.error(f"No valid data found in {data_path}")
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=['subject', 'activity', 'timestamp', 'x', 'y', 'z'])
    logging.info(f"Loaded {len(df)} rows of {sensor_type} data")

    return df

def check_data_quality(df, sensor_type):
    """
    Check data quality and report statistics
    """
    logging.info(f"\n--- {sensor_type} Data Quality Report ---")
    logging.info(f"Shape: {df.shape}")
    logging.info(f"Missing values: {df.isnull().sum().sum()}")

    # Check for outliers using z-score
    z_scores = stats.zscore(df[['x', 'y', 'z']])
    outliers = (abs(z_scores) > 3).any(axis=1)
    logging.info(f"Potential outliers (|z| > 3): {outliers.sum()} ({outliers.sum()/len(df)*100:.2f}%)")

    # Activity distribution
    activity_counts = df['activity'].value_counts()
    logging.info(f"Activity distribution:\n{activity_counts}")

    # Subject distribution
    subject_counts = df['subject'].value_counts()
    logging.info(f"Number of subjects: {len(subject_counts)}")

    return outliers

def merge_sensor_data(accel_df, gyro_df, time_tolerance_ms=100):
    """
    Merge accelerometer and gyroscope data with time tolerance

    Parameters:
    accel_df (pd.DataFrame): Accelerometer data
    gyro_df (pd.DataFrame): Gyroscope data
    time_tolerance_ms (int): Time tolerance in milliseconds for matching timestamps

    Returns:
    pd.DataFrame: Merged sensor data
    """
    logging.info("Merging accelerometer and gyroscope data")

    # Create time windows for matching
    accel_df['timestamp_min'] = accel_df['timestamp'] - time_tolerance_ms
    accel_df['timestamp_max'] = accel_df['timestamp'] + time_tolerance_ms

    # Prepare for merge
    accel_merge = accel_df.copy()
    gyro_merge = gyro_df.copy()

    # Rename columns for clarity after merge
    accel_merge.rename(columns={'x': 'x_accel', 'y': 'y_accel', 'z': 'z_accel'}, inplace=True)
    gyro_merge.rename(columns={'x': 'x_gyro', 'y': 'y_gyro', 'z': 'z_gyro'}, inplace=True)

    # Perform merge with time tolerance
    merged = pd.merge_asof(
        accel_merge.sort_values('timestamp'),
        gyro_merge.sort_values('timestamp'),
        on='timestamp',
        by=['subject', 'activity'],
        tolerance=time_tolerance_ms,
        direction='nearest'
    )

    # Drop auxiliary columns
    merged.drop(['timestamp_min', 'timestamp_max'], axis=1, inplace=True)

    # Check for missing values after merge
    missing_values = merged.isnull().sum()
    if missing_values.sum() > 0:
        logging.warning(f"Missing values after merge:\n{missing_values[missing_values > 0]}")

        # Fill missing values or drop rows as appropriate
        for col in ['x_gyro', 'y_gyro', 'z_gyro']:
            if missing_values[col] > 0:
                # Fill missing values with the median for that subject/activity
                merged[col] = merged.groupby(['subject', 'activity'])[col].transform(
                    lambda x: x.fillna(x.median())
                )

        # Drop any remaining rows with missing values
        before_drop = len(merged)
        merged.dropna(inplace=True)
        after_drop = len(merged)
        if before_drop > after_drop:
            logging.warning(f"Dropped {before_drop - after_drop} rows with missing values")

    logging.info(f"Merged data shape: {merged.shape}")
    return merged

def extract_features_from_window(window, sensor_features):
    """
    Extract statistical features from a time window

    Parameters:
    window (pd.DataFrame): Window of sensor data
    sensor_features (list): List of sensor feature columns

    Returns:
    np.array: Array of extracted features
    """
    # Raw signal (reshape to match expected dimension)
    features = window[sensor_features].values

    return features

def create_segments_with_features(df, sensor_features, window_size=200, step_size=100):
    """
    Create time segments with feature extraction

    Parameters:
    df (pd.DataFrame): Sensor data
    sensor_features (list): List of sensor feature columns
    window_size (int): Window size in samples
    step_size (int): Step size in samples

    Returns:
    tuple: (segments, labels, subjects)
    """
    segments, labels, subjects = [], [], []
    total_windows = (len(df) - window_size) // step_size + 1

    logging.info(f"Creating segments with window size {window_size} and step size {step_size}")
    logging.info(f"Expected number of segments: ~{total_windows}")

    for i in range(0, len(df) - window_size, step_size):
        window = df.iloc[i:i+window_size]

        # Check if window contains data from a single activity and subject
        if len(window['activity'].unique()) == 1 and len(window['subject'].unique()) == 1:
            # Extract features from the window
            segment = extract_features_from_window(window, sensor_features)

            segments.append(segment)
            labels.append(window['activity'].iloc[0])
            subjects.append(window['subject'].iloc[0])

    segments = np.array(segments)
    labels = np.array(labels)
    subjects = np.array(subjects)

    logging.info(f"Created {len(segments)} segments")
    logging.info(f"Segments shape: {segments.shape}")

    return segments, labels, subjects

def split_data_by_subject(segments, labels, subjects, test_size=0.2, random_state=42):
    """
    Split data ensuring subjects in test set don't appear in training set

    Parameters:
    segments (np.array): Segmented sensor data
    labels (np.array): Activity labels
    subjects (np.array): Subject IDs
    test_size (float): Proportion of data for testing
    random_state (int): Random seed

    Returns:
    tuple: (X_train, X_test, y_train, y_test)
    """
    unique_subjects = np.unique(subjects)
    n_test_subjects = max(1, int(len(unique_subjects) * test_size))

    # Randomly select subjects for test set
    np.random.seed(random_state)
    test_subjects = np.random.choice(unique_subjects, n_test_subjects, replace=False)

    # Create masks for train/test split
    test_mask = np.isin(subjects, test_subjects)
    train_mask = ~test_mask

    X_train, X_test = segments[train_mask], segments[test_mask]
    y_train, y_test = labels[train_mask], labels[test_mask]

    logging.info(f"Split data by subject: {len(unique_subjects)} subjects total")
    logging.info(f"Training subjects: {len(unique_subjects) - n_test_subjects}")
    logging.info(f"Test subjects: {n_test_subjects}")

    return X_train, X_test, y_train, y_test

def preprocessing_pipeline():
    """
    Complete preprocessing pipeline for WISDM dataset
    """
    # 1. Load data
    accel_df = load_wisdm_data(accel_path, 'accelerometer')
    gyro_df = load_wisdm_data(gyro_path, 'gyroscope')

    if accel_df.empty or gyro_df.empty:
        logging.error("Data loading failed. Check paths and file formats.")
        return None, None, None, None, None, None, None

    # 2. Check data quality
    accel_outliers = check_data_quality(accel_df, 'Accelerometer')
    gyro_outliers = check_data_quality(gyro_df, 'Gyroscope')

    # Optional: Visualize distribution of sensor readings
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(['x', 'y', 'z']):
        plt.subplot(1, 3, i+1)
        sns.kdeplot(data=accel_df, x=col, hue='activity', common_norm=False)
        plt.title(f'Accelerometer {col} distribution by activity')
    plt.tight_layout()
    plt.savefig('accel_distribution.png')
    plt.close()

    # 3. Merge accelerometer and gyroscope data
    merged_df = merge_sensor_data(accel_df, gyro_df)

    # 4. Encode activity labels
    label_encoder = LabelEncoder()
    activity_encoded = label_encoder.fit_transform(merged_df['activity'])
    merged_df['activity'] = activity_encoded

    # Store mapping for later reference
    activity_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    logging.info(f"Activity mapping: {activity_mapping}")

    # 5. Normalize sensor features separately for accelerometer and gyroscope
    sensor_features = ['x_accel', 'y_accel', 'z_accel', 'x_gyro', 'y_gyro', 'z_gyro']
    scaler = StandardScaler()
    merged_df[sensor_features] = scaler.fit_transform(merged_df[sensor_features])

    # 6. Create segments
    WINDOW_SIZE = 200  # 10 seconds at 20 Hz
    STEP_SIZE = 100    # 50% overlap
    segments, labels, subjects = create_segments_with_features(
        merged_df, sensor_features, WINDOW_SIZE, STEP_SIZE
    )

    # 7. Split data - option 1: random split
    X_train, X_test, y_train, y_test = train_test_split(
        segments, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 7. Split data - option 2: split by subject (uncomment to use)
    # X_train, X_test, y_train, y_test = split_data_by_subject(
    #     segments, labels, subjects, test_size=0.2, random_state=42
    # )

    # 8. Visualize class distribution
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.countplot(x=y_train)
    plt.title('Class Distribution - Training Set')
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    sns.countplot(x=y_test)
    plt.title('Class Distribution - Test Set')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.close()

    # 9. Convert labels to categorical for multi-class classification if needed
    num_classes = len(np.unique(labels))
    y_train_categorical = to_categorical(y_train, num_classes=num_classes)
    y_test_categorical = to_categorical(y_test, num_classes=num_classes)

    logging.info("Preprocessing completed successfully")
    logging.info(f"X_train shape: {X_train.shape}")
    logging.info(f"y_train shape: {y_train.shape}")
    logging.info(f"X_test shape: {X_test.shape}")
    logging.info(f"y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test, y_train_categorical, y_test_categorical, activity_mapping

if __name__ == "__main__":
    # Run the preprocessing pipeline
    X_train, X_test, y_train, y_test, y_train_cat, y_test_cat, activity_mapping = preprocessing_pipeline()

    # Save preprocessed data if needed
    if X_train is not None:
        np.save('X_train.npy', X_train)
        np.save('X_test.npy', X_test)
        np.save('y_train.npy', y_train)
        np.save('y_test.npy', y_test)
        np.save('y_train_cat.npy', y_train_cat)
        np.save('y_test_cat.npy', y_test_cat)

        # Save activity mapping
        with open('activity_mapping.txt', 'w') as f:
            for idx, activity in activity_mapping.items():
                f.write(f"{idx}: {activity}\n")

        logging.info("Saved preprocessed data to disk")