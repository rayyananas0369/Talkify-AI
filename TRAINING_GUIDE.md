# Talkify AI - Training Guide

This guide explains how to organize your data and train the AI models.

## 1. Directory Structure

I have automatically created the `data` folder for you in `backend/`. You need to populate it with images/videos as follows:

### Sign Language (Image Classification)
Structure your data like this (standard Keras format):

```
backend/
├── data/
│   ├── sign_language/
│   │   ├── train/
│   │   │   ├── A/           # Put all images for 'A' gesture here
│   │   │   │   ├── img1.jpg
│   │   │   │   └── ...
│   │   │   ├── B/           # Put all images for 'B' gesture here
│   │   │   └── ...
│   │   └── val/             # (Optional) Validation set
│   │       ├── A/
│   │       └── ...
```

**Where to get data?**
- [ASL Alphabet Dataset (Kaggle)](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Automated Option**: Run the provided script in the `backend` folder:
  ```powershell
  cd backend
  python download_dataset.py
  ```
  *Note: This requires a Kaggle API key. Follow the instructions in the script.*
- **Manual Option**: Download the dataset, extract it, and copy the folders (A, B, C...) into `backend/data/sign_language/train`.

### Lip Reading (Video Classification)
Lip reading is more complex. The current script expects processed data, but generally, you would organize videos by word/sentence.

```
backend/
├── data/
│   ├── lip_reading/
│   │   ├── train/
│   │   │   ├── word_hello/
│   │   │   │   ├── video1.mp4
│   │   │   └── ...
```

**Where to get data?**
- [LipNet Dataset](https://github.com/rizkiarm/LipNet)
- [GRID Corpus](http://spandh.dcs.shef.ac.uk/gridcorpus/)

## 2. Running Training

Once you have put the images in the folders:

**Step 1**: Open your terminal in the `backend` directory.
```powershell
cd backend
```

**Step 2**: Run the training command.
```powershell
python train.py --model sign --epochs 20
```

The script will:
1. Detect images in `data/sign_language/train`.
2. Automatically count the number of classes (A, B, C...).
3. Train the model.
4. Save the result as `sign_model.h5`.

**Step 3**: Restart Backend.
After training, restart the FastAPI server so it loads the new `sign_model.h5`.
```powershell
uvicorn main:app --reload
```
