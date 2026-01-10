import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from backend.hand_tracking.mediapipe_hand import HandTracker
from backend.preprocessing.hand_keypoints import landmarks_to_list, extract_hand_landmarks

# Config
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/sign_language/alphabet/Gesture Image Data')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/sign_model.h5')
CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_']
SEQUENCE_LENGTH = 15
FEATURE_DIM = 63  # 21 landmarks * 3 (x, y, z)

def load_data():
    # Use static_image_mode=True for training on independent files
    import mediapipe.python.solutions.hands as mp_hands
    tracker_model = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )
    X = []
    y = []
    
    print(f"Loading data from: {DATA_PATH}")
    
    for idx, cls in enumerate(CLASSES):
        cls_path = os.path.join(DATA_PATH, cls)
        if not os.path.exists(cls_path):
            continue
            
        print(f"Processing class {idx+1}/{len(CLASSES)}: {cls}")
        files = [f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.png'))]
        
        current_seq = []
        count = 0
        for f in files[:200]:  # Reduced to 200 per class for speed
            img = cv2.imread(os.path.join(cls_path, f))
            if img is None: continue
            
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = tracker_model.process(rgb)
            
            if res.multi_hand_landmarks:
                landmarks = extract_hand_landmarks(res)
                feat = landmarks_to_list(landmarks)
                current_seq.append(feat)
                
                if len(current_seq) == SEQUENCE_LENGTH:
                    X.append(current_seq)
                    y.append(idx)
                    current_seq = []
                    count += 1
            
        print(f"  -> Generated {count} sequences for {cls}")
                
    print(f"\nTotal samples loaded: {len(X)}")
    return np.array(X), np.array(y)

def create_model(num_classes):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, FEATURE_DIM)),
        Dropout(0.2),
        LSTM(128, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    print("=" * 60)
    print("ASL Alphabet Sign Language Training with MediaPipe Landmarks")
    print("=" * 60)
    
    # Load data
    X, y = load_data()
    
    if len(X) == 0:
        print("Error: No data loaded. Please check the dataset path and ensure hands are detected.")
        exit(1)
        
    # Convert labels to categorical
    y = to_categorical(y, num_classes=len(CLASSES))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nDataset summary: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Build and train model
    print("\nBuilding model...")
    model = create_model(len(CLASSES))
    model.summary()
    
    print("\nStarting training with MediaPipe landmarks...")
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
