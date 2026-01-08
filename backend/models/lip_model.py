import tensorflow as tf
from tensorflow.keras import layers, models

def create_lip_model(num_classes, sequence_length=15):
    # CNN part for spatial features
    cnn = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(50, 100, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten()
    ])

    # LSTM part for temporal features
    model = models.Sequential([
        layers.TimeDistributed(cnn, input_shape=(sequence_length, 50, 100, 3)),
        layers.LSTM(128, return_sequences=False),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
