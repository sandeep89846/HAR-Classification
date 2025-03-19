import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import matplotlib.widgets as widgets
from matplotlib.gridspec import GridSpec
import random

# Load the data
csv_file = r"C:\Users\gontu\OneDrive\Documents\HAR Classification\concatenated_unextrapolated.csv"
df = pd.read_csv(csv_file)

# Ensure timestamp is in int64 (milliseconds)
df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce').astype('int64')

# Check duplicates before grouping
print(f"Before cleaning: {len(df)} duplicate timestamp-activity pairs: {df.duplicated(subset=['subject', 'activity', 'timestamp']).sum()}")

# Group by (subject, activity, timestamp) and average x, y, z
df = df.groupby(['subject', 'activity', 'timestamp'], as_index=False).agg({'x': 'mean', 'y': 'mean', 'z': 'mean'})

print(f"After cleaning: {len(df)}")

# Save the cleaned dataset
cleaned_csv_file = r"C:\Users\gontu\OneDrive\Documents\HAR Classification\concatenated_cleaned.csv"
df.to_csv(cleaned_csv_file, index=False)

print(f"Saved cleaned dataset to {cleaned_csv_file}")
