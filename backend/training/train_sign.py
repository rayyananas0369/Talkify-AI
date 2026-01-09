import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Config
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/sign_language/alphabet/Gesture Image Data')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/sign_model.h5')
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
SEQUENCE_LENGTH = 15
IMG_SIZE = 64
FEATURE_DIM = 128  # Flattened image features

def extract_features(image_path):
    """Extract features from image using simple CNN-like preprocessing"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Resize and normalize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype('float32') / 255.0
    
    # Simple feature extraction: divide into grid and get mean values
    grid_size = 8
    cell_h = IMG_SIZE // grid_size
    cell_w = IMG_SIZE // grid_size
    features = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            cell = img[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            features.append(np.mean(cell))
            features.append(np.std(cell))
    
    return np.array(features)

def load_data():
    """Load and preprocess the ASL Alphabet dataset"""
    sequences = []
    labels = []
    
    print(f"Loading data from: {DATA_PATH}")
    
    for idx, class_name in enumerate(CLASSES):
        class_path = os.path.join(DATA_PATH, class_name)
        
        if not os.path.exists(class_path):
            print(f"Warning: Class folder '{class_name}' not found, skipping...")
            continue
        
        image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # Limit to 100 images per class for faster training
        image_files = image_files[:100]
        
        print(f"Processing class '{class_name}': {len(image_files)} images")
        
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            features = extract_features(img_path)
            
            if features is not None:
                # Create sequence by repeating the features (simulating temporal data)
                sequence = np.array([features] * SEQUENCE_LENGTH)
                sequences.append(sequence)
                labels.append(idx)
    
    print(f"\nTotal samples loaded: {len(sequences)}")
    print(f"Number of classes: {len(CLASSES)}")
    
    return np.array(sequences), np.array(labels)

def build_model():
    """Build LSTM model for sign language recognition"""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(SEQUENCE_LENGTH, FEATURE_DIM)),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(len(CLASSES), activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    print("=" * 60)
    print("ASL Alphabet Sign Language Training")
    print("=" * 60)
    
    # Load data
    X, y = load_data()
    
    if len(X) == 0:
        print("ERROR: No data loaded. Please check the dataset path.")
        exit(1)
    
    # Convert labels to categorical
    y = to_categorical(y, num_classes=len(CLASSES))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Build and train model
    print("\nBuilding model...")
    model = build_model()
    model.summary()
    
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
