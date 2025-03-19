import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load the extracted features
features_df = pd.read_csv('wisdm_accel_ABCDE_features.csv')
activities_df = pd.read_csv('wisdm_activity_accel_ABCDE_features.csv')

# Assuming the last column is the activity label
X = features_df.iloc[:, :-1].values  # All columns except the last one
y = activities_df.iloc[:, -1].values  # Only the last column

# Encode the activity labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(np.unique(y_encoded))
y_categorical = to_categorical(y_encoded)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
)

# Reshape input to be 3D [samples, time steps, features] for LSTM
# For this implementation, we'll treat each window as a single time step with multiple features
# If you have temporal sequence data, you might need to adjust this reshape
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define callbacks for early stopping and model checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
model_checkpoint = ModelCheckpoint('best_lstm_har_model.h5', monitor='val_accuracy', 
                                  save_best_only=True, verbose=1, mode='max')

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('lstm_training_history.png')
plt.show()

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Map back to original activity labels
y_pred_labels = label_encoder.inverse_transform(y_pred_classes)
y_true_labels = label_encoder.inverse_transform(y_true_classes)

# Create a confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_true_classes, y_pred_classes)
class_names = label_encoder.classes_

# Plot confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Add text annotations to the confusion matrix
thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig('confusion_matrix_1.png')
plt.show()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

# Function to predict activity for new data
def predict_activity(new_data, model, scaler, label_encoder):
    """
    Predict activity for new accelerometer data features
    
    Parameters:
    new_data (DataFrame or array): Extracted features in the same format as training data
    model: Trained LSTM model
    scaler: Fitted StandardScaler
    label_encoder: Fitted LabelEncoder
    
    Returns:
    str: Predicted activity label
    """
    # Scale the features
    if isinstance(new_data, pd.DataFrame):
        new_data = new_data.values
    
    new_data_scaled = scaler.transform(new_data.reshape(1, -1))
    
    # Reshape for LSTM
    new_data_reshaped = new_data_scaled.reshape(1, 1, new_data_scaled.shape[1])
    
    # Predict
    prediction = model.predict(new_data_reshaped)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    # Map back to original label
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    
    return predicted_label

# Save model and preprocessing objects for later use
import joblib
joblib.dump(scaler, 'har_feature_scaler.pkl')
joblib.dump(label_encoder, 'har_label_encoder.pkl')
# The model is already saved by ModelCheckpoint callback