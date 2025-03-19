import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the processed data
processed_data_path = "processed_data.csv"

if not os.path.exists(processed_data_path):
    print(f"Error: {processed_data_path} not found. Please run the data processing script first.")
    exit()

# Load the data
df = pd.read_csv(processed_data_path)

# Create a directory for the plots
os.makedirs('activity_plots', exist_ok=True)

# Define activity labels (map activity codes to names if needed)
activity_labels = {
    'A': 'Walking',
    'B': 'Jogging',
    'C': 'Stairs',
    'D': 'Sitting',
    'E': 'Standing'
}

# Set the style for the plots
sns.set(style="whitegrid")
plt.figure(figsize=(14, 10))

# Function to create and save bar plots
def create_activity_bar_plots():
    # Get aggregated stats for each activity
    activity_stats = df.groupby('activity').agg({
        'x': ['mean', 'std', 'min', 'max'],
        'y': ['mean', 'std', 'min', 'max'],
        'z': ['mean', 'std', 'min', 'max']
    })
    
    # Reset index to make 'activity' a column
    activity_stats = activity_stats.reset_index()
    
    # Plot 1: Mean values for each activity across all axes
    plt.figure(figsize=(12, 8))
    
    # Create a DataFrame with the mean values for easier plotting
    mean_data = pd.DataFrame({
        'Activity': [activity_labels.get(act, act) for act in activity_stats['activity']],
        'X-axis': activity_stats['x']['mean'],
        'Y-axis': activity_stats['y']['mean'],
        'Z-axis': activity_stats['z']['mean']
    })
    
    # Melt the DataFrame for easier plotting with seaborn
    mean_data_melted = pd.melt(mean_data, id_vars=['Activity'], 
                               var_name='Axis', value_name='Mean Acceleration')
    
    # Create the plot
    ax = sns.barplot(x='Activity', y='Mean Acceleration', hue='Axis', data=mean_data_melted)
    
    plt.title('Mean Acceleration by Activity and Axis', fontsize=16)
    plt.xlabel('Activity', fontsize=14)
    plt.ylabel('Mean Acceleration (m/s²)', fontsize=14)
    plt.legend(title='Axis')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=0)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('activity_plots/mean_acceleration_by_activity.png', dpi=300)
    plt.close()
    
    # Plot 2: Standard deviation for each activity across all axes
    plt.figure(figsize=(12, 8))
    
    # Create a DataFrame with the std values
    std_data = pd.DataFrame({
        'Activity': [activity_labels.get(act, act) for act in activity_stats['activity']],
        'X-axis': activity_stats['x']['std'],
        'Y-axis': activity_stats['y']['std'],
        'Z-axis': activity_stats['z']['std']
    })
    
    # Melt the DataFrame
    std_data_melted = pd.melt(std_data, id_vars=['Activity'], 
                              var_name='Axis', value_name='Standard Deviation')
    
    # Create the plot
    ax = sns.barplot(x='Activity', y='Standard Deviation', hue='Axis', data=std_data_melted)
    
    plt.title('Acceleration Variability by Activity and Axis', fontsize=16)
    plt.xlabel('Activity', fontsize=14)
    plt.ylabel('Standard Deviation of Acceleration', fontsize=14)
    plt.legend(title='Axis')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('activity_plots/acceleration_variability_by_activity.png', dpi=300)
    plt.close()
    
    # Plot 3: Range (max-min) for each activity across all axes
    plt.figure(figsize=(12, 8))
    
    # Calculate range
    range_data = pd.DataFrame({
        'Activity': [activity_labels.get(act, act) for act in activity_stats['activity']],
        'X-axis': activity_stats['x']['max'] - activity_stats['x']['min'],
        'Y-axis': activity_stats['y']['max'] - activity_stats['y']['min'],
        'Z-axis': activity_stats['z']['max'] - activity_stats['z']['min']
    })
    
    # Melt the DataFrame
    range_data_melted = pd.melt(range_data, id_vars=['Activity'], 
                               var_name='Axis', value_name='Acceleration Range')
    
    # Create the plot
    ax = sns.barplot(x='Activity', y='Acceleration Range', hue='Axis', data=range_data_melted)
    
    plt.title('Acceleration Range by Activity and Axis', fontsize=16)
    plt.xlabel('Activity', fontsize=14)
    plt.ylabel('Acceleration Range (max-min)', fontsize=14)
    plt.legend(title='Axis')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('activity_plots/acceleration_range_by_activity.png', dpi=300)
    plt.close()
    
    # Plot 4: Individual user plots by activity
    # Get unique subjects
    subjects = df['subject'].unique()
    
    # For each activity, create a plot showing the mean acceleration for each user
    for activity in df['activity'].unique():
        activity_name = activity_labels.get(activity, activity)
        
        plt.figure(figsize=(14, 10))
        
        # Filter data for this activity
        activity_data = df[df['activity'] == activity]
        
        # Aggregate by subject
        subject_stats = activity_data.groupby('subject').agg({
            'x': 'mean',
            'y': 'mean',
            'z': 'mean'
        }).reset_index()
        
        # Create the plot
        subject_stats_melted = pd.melt(subject_stats, id_vars=['subject'], 
                                      var_name='Axis', value_name='Mean Acceleration')
        
        ax = sns.barplot(x='subject', y='Mean Acceleration', hue='Axis', data=subject_stats_melted)
        
        plt.title(f'Mean Acceleration by User for Activity: {activity_name}', fontsize=16)
        plt.xlabel('User ID', fontsize=14)
        plt.ylabel('Mean Acceleration (m/s²)', fontsize=14)
        plt.legend(title='Axis')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(f'activity_plots/mean_acceleration_activity_{activity}_by_user.png', dpi=300)
        plt.close()

# Execute the function
if __name__ == "__main__":
    print("Creating activity bar plots...")
    create_activity_bar_plots()
    print("Plots created and saved to 'activity_plots' directory.")