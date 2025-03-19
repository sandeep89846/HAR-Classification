import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load preprocessed data
def load_preprocessed_data(path='.'):
    """
    Load preprocessed data from saved files
    """
    try:
        X_train = np.load(f'{path}/X_train.npy')
        X_test = np.load(f'{path}/X_test.npy')
        y_train = np.load(f'{path}/y_train.npy')
        y_test = np.load(f'{path}/y_test.npy')
        y_train_cat = np.load(f'{path}/y_train_cat.npy')
        y_test_cat = np.load(f'{path}/y_test_cat.npy')

        # Load activity mapping
        activity_mapping = {}
        try:
            with open(f'{path}/activity_mapping.txt', 'r') as f:
                for line in f:
                    idx, activity = line.strip().split(': ')
                    activity_mapping[int(idx)] = activity
        except FileNotFoundError:
            logging.warning("Activity mapping file not found. Using numerical labels instead.")
            # Create a default mapping using numerical indices
            unique_labels = np.unique(np.concatenate([y_train, y_test]))
            activity_mapping = {idx: f"Activity_{idx}" for idx in unique_labels}
            
            # Save the activity mapping for future use
            with open(f'{path}/activity_mapping.txt', 'w') as f:
                for idx, activity in activity_mapping.items():
                    f.write(f"{idx}: {activity}\n")
            logging.info("Created and saved default activity mapping")

        logging.info("Successfully loaded preprocessed data")
        return X_train, X_test, y_train, y_test, y_train_cat, y_test_cat, activity_mapping
    except Exception as e:
        logging.error(f"Error loading preprocessed data: {e}")
        return None, None, None, None, None, None, None

def build_cnn_lstm_model(input_shape, num_classes):
    """
    Build a CNN-LSTM hybrid model for HAR classification

    Parameters:
    input_shape (tuple): Shape of input data (window_size, features)
    num_classes (int): Number of activity classes

    Returns:
    tensorflow.keras.Model: Compiled model
    """
    model = Sequential([
        # CNN layers for feature extraction
        Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.4),

        # LSTM layer for sequence modeling
        LSTM(128, return_sequences=False),
        Dropout(0.5),

        # Dense layers for classification
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    # Compile model with Adam optimizer
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    logging.info(f"Model summary: {model.summary()}")
    return model

def build_multi_head_cnn_model(input_shape, num_classes):
    """
    Build a multi-head CNN model for HAR classification
    This model separates processing for accelerometer and gyroscope data

    Parameters:
    input_shape (tuple): Shape of input data (window_size, features)
    num_classes (int): Number of activity classes

    Returns:
    tensorflow.keras.Model: Compiled model
    """
    # Input layer
    input_layer = tf.keras.layers.Input(shape=input_shape)

    # Split input into accelerometer and gyroscope channels
    # Assuming first 3 channels are accelerometer (x,y,z) and last 3 are gyroscope (x,y,z)
    accel_input = tf.keras.layers.Lambda(lambda x: x[:, :, 0:3])(input_layer)
    gyro_input = tf.keras.layers.Lambda(lambda x: x[:, :, 3:6])(input_layer)

    # Accelerometer branch
    accel_conv1 = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(accel_input)
    accel_bn1 = BatchNormalization()(accel_conv1)
    accel_pool1 = MaxPooling1D(pool_size=2)(accel_bn1)

    accel_conv2 = Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(accel_pool1)
    accel_bn2 = BatchNormalization()(accel_conv2)
    accel_pool2 = MaxPooling1D(pool_size=2)(accel_bn2)

    # Gyroscope branch
    gyro_conv1 = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(gyro_input)
    gyro_bn1 = BatchNormalization()(gyro_conv1)
    gyro_pool1 = MaxPooling1D(pool_size=2)(gyro_bn1)

    gyro_conv2 = Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(gyro_pool1)
    gyro_bn2 = BatchNormalization()(gyro_conv2)
    gyro_pool2 = MaxPooling1D(pool_size=2)(gyro_bn2)

    # Merge branches
    merged = tf.keras.layers.concatenate([accel_pool2, gyro_pool2])

    # Shared layers after merge
    conv3 = Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(merged)
    bn3 = BatchNormalization()(conv3)
    pool3 = MaxPooling1D(pool_size=2)(bn3)
    dropout1 = Dropout(0.4)(pool3)

    # Flatten and dense layers
    flat = Flatten()(dropout1)
    dense1 = Dense(128, activation='relu')(flat)
    bn4 = BatchNormalization()(dense1)
    dropout2 = Dropout(0.5)(bn4)
    output = Dense(num_classes, activation='softmax')(dropout2)

    # Define and compile model
    model = tf.keras.models.Model(inputs=input_layer, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    logging.info(f"Model summary: {model.summary()}")
    return model

def train_model(model, X_train, y_train_cat, X_test, y_test_cat, batch_size=32, epochs=100):
    """
    Train the HAR classification model with early stopping and learning rate reduction

    Parameters:
    model (tensorflow.keras.Model): Model to train
    X_train, y_train_cat, X_test, y_test_cat: Training and testing data
    batch_size (int): Batch size for training
    epochs (int): Maximum number of epochs

    Returns:
    tuple: (trained_model, history)
    """
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            'har_model_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # Train model
    logging.info("Starting model training...")
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    logging.info("Model training completed")
    return model, history

def evaluate_model(model, X_test, y_test, y_test_cat, activity_mapping, history):
    """
    Evaluate the trained model and visualize results

    Parameters:
    model (tensorflow.keras.Model): Trained model
    X_test, y_test, y_test_cat: Test data
    activity_mapping (dict): Mapping from class indices to activity names
    history: Training history from model.fit()

    Returns:
    pd.DataFrame: DataFrame containing true and predicted labels for error analysis
    """
    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test_cat)
    logging.info(f"Test Loss: {loss:.4f}")
    logging.info(f"Test Accuracy: {accuracy:.4f}")

    # Get predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Classification report
    class_names = [activity_mapping[i] for i in sorted(activity_mapping.keys())]
    report = classification_report(y_test, y_pred, target_names=class_names)
    logging.info("Classification Report:\n" + report)

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()  # Close the figure to free memory

    # Plot accuracy and loss curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()  # Close the figure to free memory

    # Analyze most common misclassifications
    error_df = pd.DataFrame({
        'true': [activity_mapping[i] for i in y_test],
        'pred': [activity_mapping[i] for i in y_pred]
    })
    errors = error_df[error_df['true'] != error_df['pred']]
    error_counts = errors.groupby(['true', 'pred']).size().reset_index(name='count')
    error_counts = error_counts.sort_values('count', ascending=False)

    logging.info("Top misclassifications:")
    logging.info(error_counts.head(10))

    return error_df

def visualize_activations(model, X_test, y_test, activity_mapping, layer_name=None):
    """
    Visualize activations of a specific layer for different activity classes

    Parameters:
    model (tensorflow.keras.Model): Trained model
    X_test, y_test: Test data
    activity_mapping (dict): Mapping from class indices to activity names
    layer_name (str): Name of the layer to visualize, if None, uses the last conv layer
    """
    if layer_name is None:
        # Find the last convolutional layer
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv1D):
                layer_name = layer.name
                break
        
        if layer_name is None:
            logging.warning("No Conv1D layer found in the model. Skipping activation visualization.")
            return

    # Create a model that will return the activations
    try:
        activation_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=model.get_layer(layer_name).output
        )
    except ValueError as e:
        logging.error(f"Error creating activation model: {e}")
        logging.info(f"Available layers: {[layer.name for layer in model.layers]}")
        return

    # Select a few examples of each class
    unique_classes = np.unique(y_test)
    n_classes = len(unique_classes)
    n_examples = min(3, len(X_test) // n_classes)  # 3 examples per class if possible

    plt.figure(figsize=(15, n_classes * 4))

    for i, class_idx in enumerate(unique_classes):
        # Get examples of this class
        class_indices = np.where(y_test == class_idx)[0]
        if len(class_indices) == 0:
            continue
            
        class_indices = class_indices[:n_examples]

        for j, idx in enumerate(class_indices):
            # Get activations
            sample = X_test[idx:idx+1]
            activations = activation_model.predict(sample)
            activations = activations[0]  # Remove batch dimension

            # Plot activations as heatmap
            plt.subplot(n_classes, n_examples, i * n_examples + j + 1)
            sns.heatmap(activations.T, cmap='viridis')
            plt.title(f"{activity_mapping[class_idx]}")
            plt.xlabel('Time')
            plt.ylabel('Filters')

    plt.tight_layout()
    plt.savefig('layer_activations.png')
    plt.close()  # Close the figure to free memory

if __name__ == "__main__":
    # Load preprocessed data
    X_train, X_test, y_train, y_test, y_train_cat, y_test_cat, activity_mapping = load_preprocessed_data()

    if X_train is None:
        logging.error("Failed to load data. Exiting.")
        exit(1)

    # Check data shape and prepare model input
    window_size, n_features = X_train.shape[1], X_train.shape[2]
    num_classes = y_train_cat.shape[1]

    logging.info(f"Window size: {window_size}")
    logging.info(f"Number of features: {n_features}")
    logging.info(f"Number of classes: {num_classes}")
    logging.info(f"Activity classes: {activity_mapping}")

    # Choose model architecture based on data characteristics
    if n_features == 6:  # 3 accel + 3 gyro features
        logging.info("Building multi-head CNN model for accelerometer and gyroscope data")
        model = build_multi_head_cnn_model((window_size, n_features), num_classes)
    else:
        logging.info("Building CNN-LSTM hybrid model")
        model = build_cnn_lstm_model((window_size, n_features), num_classes)

    # Train model
    model, history = train_model(model, X_train, y_train_cat, X_test, y_test_cat)

    # Evaluate model
    error_df = evaluate_model(model, X_test, y_test, y_test_cat, activity_mapping, history)

    # Visualize layer activations
    visualize_activations(model, X_test, y_test, activity_mapping)

    # Save model
    model.save('har_model_final.h5')
    logging.info("Model saved to har_model_final.h5")