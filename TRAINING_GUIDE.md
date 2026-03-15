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

### Voice Recognition (Video Classification)
Voice recognition is more complex. The current script expects processed data, but generally, you would organize videos by word/sentence.

```
backend/
├── data/
│   ├── voice_reading/
│   │   ├── train/
│   │   │   ├── word_hello/
│   │   │   │   ├── video1.mp4
│   │   │   └── ...
```

**Where to get data?**
- [VoiceNet Dataset](https://github.com/rizkiarm/VoiceNet)
- [GRID Corpus](http://spandh.dcs.shef.ac.uk/gridcorpus/)

## 2. Running Training

Once you have put the images in the folders:

**Step 1**: Open your terminal in the `backend` directory.
```powershell
cd backend
```

**Step 2 (Local)**: Run the training command on your PC.
```powershell
python training/train_sign.py
```

**Step 2 (Cloud - RECOMMENDED)**:
1. Upload the project to Google Drive.
2. Open `docs/Talkify_AI_Colab_Training.ipynb` in [Google Colab](https://colab.research.google.com/).
3. Follow the notebook instructions to train using a GPU.
4. Download the resulting `sign_model_colab.h5` and place it in `backend/models/sign_model.h5`.

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
