import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load the saved model and scaler
model_path = r"C:\Users\gontu\OneDrive\Documents\HAR Classification\xgboost_model.pkl"
scaler_path = r"C:\Users\gontu\OneDrive\Documents\HAR Classification\scaler.pkl"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Load the original dataset to understand its structure
original_data_path = r"C:\Users\gontu\OneDrive\Documents\HAR Classification\wisdm_accel_ABCDE_features.csv"
original_df = pd.read_csv(original_data_path)
print("Original dataset columns:", original_df.columns.tolist())

# Load the new dataset
new_data_path = r"C:\Users\gontu\OneDrive\Documents\HAR Classification\ku_har_activity_accel_ABCDE_features.csv"
new_df = pd.read_csv(new_data_path)
print("New dataset columns:", new_df.columns.tolist())

# Look at the first few rows of the new dataset
print("\nNew dataset sample:")
print(new_df.head(2))

# Check if there's an 'activity' column in the new dataset
has_activity = 'activity' in new_df.columns
print(f"Does new dataset have 'activity' column: {has_activity}")

# Create a new LabelEncoder for the activity column
label_encoder = LabelEncoder()

# If the original dataset has alphabetic labels, fit the encoder to those
if has_activity:
    # Get unique activities from both datasets
    original_activities = original_df['activity'].unique()
    new_activities = new_df['activity'].unique()
    all_activities = np.unique(np.concatenate([original_activities, new_activities]))
    
    # Fit the encoder with all possible activities
    label_encoder.fit(all_activities)
    
    # Transform the target column
    y_true = label_encoder.transform(new_df['activity'])
    print("Unique activities in new dataset:", new_df['activity'].unique())
    print("Encoded activities:", label_encoder.transform(new_df['activity'].unique()))

# Get feature columns from the original dataset
original_features = [col for col in original_df.columns if col != 'activity']
print(f"Original features: {original_features}")

# Prepare features for the new dataset
X_new = pd.DataFrame(index=new_df.index)

# Ensure the new dataset has the same features as the original
for feature in original_features:
    if feature in new_df.columns:
        X_new[feature] = new_df[feature]
    elif feature == 'subject' and 'subject' not in new_df.columns:
        # Add dummy subject column if it's missing
        print(f"Adding dummy 'subject' column")
        X_new['subject'] = 0
    else:
        print(f"Missing feature: {feature}, filling with zeros")
        X_new[feature] = 0

print(f"Number of features in prepared new dataset: {X_new.shape[1]}")
print(f"Features match original dataset: {set(X_new.columns) == set(original_features)}")

# Scale the features
try:
    X_new_scaled = scaler.transform(X_new)
    print("Scaling successful")
except Exception as e:
    print(f"Scaling error: {e}")
    print("Feature mismatch between training and new dataset!")
    print(f"Original features: {original_features}")
    print(f"New features: {X_new.columns.tolist()}")
    # Sort features to match original dataset
    X_new = X_new[original_features]
    X_new_scaled = scaler.transform(X_new)
    print("Scaling successful after reordering features")

# Make predictions
y_pred_numeric = model.predict(X_new_scaled)

# If we have actual activity labels in the new dataset, evaluate
if has_activity:
    # Convert numeric predictions back to original labels
    y_pred_labels = label_encoder.inverse_transform(y_pred_numeric.astype(int))
    y_true_labels = new_df['activity']
    
    # Calculate metrics
    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    f1 = f1_score(y_true, y_pred_numeric, average='weighted')
    
    print(f"\nAccuracy on new dataset: {accuracy:.4f}")
    print(f"F1 Score on new dataset: {f1:.4f}")
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_true_labels, y_pred_labels))
    
    # Create confusion matrix
    conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    unique_classes = np.unique(np.concatenate([y_true_labels, y_pred_labels]))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=unique_classes,
                yticklabels=unique_classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("Confusion matrix saved as 'confusion_matrix.png'")

# If there's no activity column, just show predictions
else:
    print("\nPredictions:")
    # Create a dataframe with predictions
    results_df = new_df.copy()
    
    # Convert numeric predictions to original labels
    results_df['predicted_activity'] = label_encoder.inverse_transform(y_pred_numeric.astype(int))
    
    # Show the first few predictions
    print(results_df[['predicted_activity']].head(10))
    
    # Save results to CSV
    results_path = r"C:\Users\gontu\OneDrive\Documents\HAR Classification\ku_har_predictions.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Saved predictions to: {results_path}")

# Feature importance
if hasattr(model, 'feature_importances_'):
    feature_importance = model.feature_importances_
    
    # Create a dataframe for better visualization
    feature_names = X_new.columns.tolist()
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(importance_df.head(10))  # Print top 10 features
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    plt.barh(importance_df['Feature'][:10], importance_df['Importance'][:10])
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importance')
    plt.gca().invert_yaxis()  # Invert y-axis to show most important at the top
    plt.savefig('feature_importance.png')
    plt.close()
    print("Feature importance plot saved as 'feature_importance.png'")

print("\nPrediction completed successfully!")