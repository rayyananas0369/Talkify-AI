# 🤝 Talkify AI: Friendly Developer Guide

Welcome to the team! This guide uses simple names and detailed comments to help everyone understand how Talkify AI works.

---

## 📂 Project Structure (Friendly Version)

| Folder Name | Simple English Name | What it does? |
| :--- | :--- | :--- |
| `backend/data_prep/` | **Data Cleaner** | Standardizes camera coordinates so they look the same for every user. |
| `backend/predictor/` | **AI Brain** | Takes cleaned data and makes the final "Sign" or "Lip" guess. |
| `backend/hand_logic/`| **Hand Finder** | Uses YOLOv8 and MediaPipe to focus ONLY on movements of hands. |
| `backend/lip_logic/` | **Lip Finder** | Focuses ONLY on the mouth area for lip reading. |
| `backend/ai_files/`  | **Memory Bank** | Stores the heavy `.h5` and `.pt` files where the AI's learning is saved. |
| `backend/extra_tools/` | **Helpful Gadgets** | Small scripts for logging errors and measuring speed (FPS). |

---

## 🗺️ Where to put the Data? (Extraction Map)

To make the "AI Brain" work, you must put the downloaded files in these specific "buckets":

1. **For ASL Alphabets (Kaggle)**:
   Extract to: `backend/data/sign_language/alphabet/`

2. **For Familiar Words (WLASL)**:
   - **Download Link**: [WLASL GitHub](https://github.com/dxli/WLASL)
   - *Note*: This is tricky! You need to clone their repo or download the `WLASL_v0.3.json`.
   - **Easier Option**: For now, you can skip this. We can train the "Sign Alphabet" first, which is easier!
   - Extract to: `backend/data/sign_language/words/`

3. **For Lip Reading (LRS3)**:
   Extract to: `backend/data/lip_reading/`

---

## 🎯 Module 2: Making Data "AI-Ready" (With Comments)

Here is the logic for **Data Cleaner**. We perform two main actions: **Normalization** (centering) and **Buffering** (memory).

### 1. Centering the Keypoints (Normalization)
*File: `data_prep/center_points.py`*

```python
def center_hand_data(landmarks):
    # ACTION: We find the "Wrist" (index 0) to use as a baseline.
    # Why? So it doesn't matter if you are at the top or bottom of the screen.
    wrist = landmarks[0]
    baseline_x, baseline_y = wrist['x'], wrist['y']
    
    clean_list = []
    for point in landmarks:
        # ACTION: Subtract the wrist position from every other finger point.
        # RESULT: The data now represents the "Shape" of the hand relative to the wrist.
        new_x = point['x'] - baseline_x
        new_y = point['y'] - baseline_y
        
        # Add to our list of 63 numbers (X, Y, Z for 21 points)
        clean_list.extend([new_x, new_y, point['z']])
        
    return clean_list
```

### 2. Creating a 15-Frame Memory (Buffering)
*File: `data_prep/save_sequence.py`*

```python
# ACTION: We store the last 15 frames of movement.
# Why? Sign language is a "Motion," not a "Picture." We need to see the path of the hand.
memory_buffer = []

def add_frame_to_memory(new_frame_data):
    # 1. Add the newest frame to our shelf
    memory_buffer.append(new_frame_data)
    
    # 2. If we have more than 15, throw away the oldest one (index 0)
    # This keeps our "Memory" fresh and fixed at 15 frames.
    if len(memory_buffer) > 15:
        memory_buffer.pop(0)
        
    # 3. Once we have exactly 15, the AI is ready to "Think"
    if len(memory_buffer) == 15:
        return True # Ready to Predict!
    return False
```

---

## ✋ Module 3: Sign Language Guessing (With Comments)

### Running the Inference
*File: `predictor/guess_sign.py`*

```python
def predict_sign_language(sequence_of_15_frames):
    # ACTION: Pass our 15 frames of memory into the AI Model file.
    # The model was trained by Nithya on thousands of ASL videos.
    raw_guess = sign_model.predict(sequence_of_15_frames)
    
    # ACTION: Find the class (A, B, C...) with the highest probability.
    best_index = find_highest_probability(raw_guess)
    confidence = raw_guess[best_index]
    
    # ACTION: Only show the user the text if the AI is very sure (> 85%).
    # This prevents the text from changing randomly due to "Noise."
    if confidence > 0.85:
        return alphabet_list[best_index]
    else:
        return "Analyzing..."
```

---
*Talkify AI Guide for Friends - 2026*
