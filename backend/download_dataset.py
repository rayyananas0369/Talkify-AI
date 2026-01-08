import os
import zipfile
import subprocess
import sys

def setup_directories():
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_dirs = [
        "data/sign_language/train",
        "data/sign_language/val",
        "data/lip_reading/train",
        "data/lip_reading/val"
    ]
    
    for d in data_dirs:
        full_path = os.path.join(base_path, d)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            print(f"Created directory: {full_path}")
        else:
            print(f"Directory already exists: {full_path}")

def download_asl_alphabet():
    print("\n--- ASL Alphabet Dataset Download ---")
    print("This requires the 'kaggle' python package and a Kaggle API key.")
    print("If you haven't set up Kaggle API yet, please follow:")
    print("1. Go to kaggle.com -> Settings -> Create New API Token.")
    print("2. Place the downloaded 'kaggle.json' in C:\\Users\\<YourUsername>\\.kaggle\\")
    
    try:
        import kaggle
    except ImportError:
        print("\n'kaggle' package not found. Installing now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        import kaggle

    dataset = "grassknoted/asl-alphabet"
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/sign_language")
    
    print(f"Downloading {dataset} to {output_path}...")
    kaggle.api.dataset_download_files(dataset, path=output_path, unzip=True)
    print("Download and extraction complete!")

if __name__ == "__main__":
    setup_directories()
    
    # We don't automatically trigger the download in case the user hasn't set up API keys.
    # But we provide the function call here.
    choice = input("\nDo you want to attempt downloading the ASL Alphabet dataset from Kaggle now? (y/n): ")
    if choice.lower() == 'y':
        try:
            download_asl_alphabet()
        except Exception as e:
            print(f"\nError downloading dataset: {e}")
            print("Make sure your kaggle.json is correctly placed.")
    else:
        print("\nSkipping automatic download. You can run this script later or download manually from:")
        print("https://www.kaggle.com/datasets/grassknoted/asl-alphabet")
