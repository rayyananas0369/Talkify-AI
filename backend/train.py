import argparse
import os
import tensorflow as tf
from models.sign_model import create_sign_model
from models.lip_model import create_lip_model

def train_sign(epochs=20):
    print("Training Sign Language Model...")
    # Placeholder for data loading logic
    # In a real scenario, use tf.keras.preprocessing.image_dataset_from_directory
    model = create_sign_model(26)
    model.summary()
    print("Run with real data to complete training.")
    # model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    model.save("sign_model_new.h5")

def train_lip(epochs=20):
    print("Training Lip Reading Model...")
    model = create_lip_model(10)
    model.summary()
    print("Run with real data to complete training.")
    # model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    model.save("lip_model_new.h5")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="sign", help="sign or lip")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    if args.model == "sign":
        train_sign(args.epochs)
    else:
        train_lip(args.epochs)
