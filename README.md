# 🎙️ VoiceVerifier-vv
A FastAPI-powered voice classification system that identifies speaker types from uploaded audio files. It leverages **SpeechBrain’s ECAPA-TDNN** model to extract speaker embeddings, removes silence using **Silero VAD**, and classifies voices using a **Random Forest** model based on **cosine similarity**.

---

## 📌 Project Summary

This project helps identify a speaker’s voice type (e.g., modern, lofi, emo) based on their audio characteristics. It includes a full pipeline for:

- Preparing and cleaning audio data
- Extracting embeddings
- Training a classifier
- Running real-time inference via API

---

## 🛠️ Tech Stack

- **Python 3.8+**
- [FastAPI](https://fastapi.tiangolo.com/) — for REST API
- [SpeechBrain](https://speechbrain.readthedocs.io/) — ECAPA-TDNN speaker encoder
- [Silero VAD](https://github.com/snakers4/silero-vad) — for silence removal
- **Scikit-learn** — for RandomForestClassifier and evaluation
- **TorchAudio** — for audio loading/saving

---

## 🏗️ Full Pipeline Overview

### 1. 🎧 **Audio Data Regrouping**
Script: `1_data-regrouping.py`

- Original dataset contains raw audio clips with various keywords.
- Files are reorganized into:
dataset/
* ├── speaker_1/ # 'modern'
- ├── speaker_2/ # 'lofi'
- ├── speaker_3/ # 'emo'
- ├──
- ├──
- ├── speaker_n/ # 'name'
  
### 2. 🔇 **Silence Removal**
Script: `2_prepare_clean_dataset.py`

- Uses **Silero VAD** to detect speech segments.
- Non-speech parts are removed.
- Output is saved in `clean_dataset/`.

### 3. 🧠 **Embedding Extraction**
Script: `3_prepare_embeddings.py`

- SpeechBrain’s ECAPA-TDNN model extracts a vector representation (embedding) of each voice.
- Embeddings and labels are saved as `.npy` files.

### 4. ✂️ **Train/Test Split**
Script: `4_train_test_split.py`

- Embeddings are split into training and test sets using stratified sampling.

### 5. 🧪 **Model Training**
Script: `5_train_model.py`

- A **Random Forest Classifier** is trained on the speaker embeddings.
- The trained model is saved as: `final_voice_classifier.pkl`.

### 6. 📊 **Evaluation**
Script: `evaluate_model.py`

- Model is evaluated on the test set.
- Reports:
- Classification Report (precision, recall, f1-score)
- Confusion Matrix

### 7. 🔍 **Inference & Prediction**
Scripts: `7_inference.py` and `voice_classifier_api.py`

- Runs inference on new audio files.
- Predicts class by comparing the embedding to class centroids using **cosine similarity**.
- If similarity < 0.6 → labeled as "unknown".

---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/Mrkomiljon/VoiceVerifier-vv.git
cd VoiceVerifier-vv
```

```
# Install dependencies
pip install -r requirements.txt
```
- Ensure you have ffmpeg installed to support all audio formats (.mp3, .flac, etc.).
## ▶️ Run FastAPI Server
```
uvicorn voice_classifier_api:app --reload
```
API will be available at:
http://127.0.0.1:8000

📡 API Endpoints
GET /
Test endpoint to check if the API is running.

POST /predict
Upload an audio file to get the predicted speaker class.

Example (using curl):
```
curl -X POST "http://127.0.0.1:8000/predict" \
  -H  "accept: application/json" \
  -H  "Content-Type: multipart/form-data" \
  -F "file=@test_audio.wav"
```
## 🧪 Example Output
![image](https://github.com/user-attachments/assets/de5ee3da-4077-405e-8151-2f2853484651)

```
{
  "status": "success",
  "predicted_class": 1,
  "similarity": 0.9178803563117981,
  "message": "이 음성은 1번 음성 유형과 일치합니다."
}
```
If similarity is below threshold:

![image](https://github.com/user-attachments/assets/93cee9ed-5e46-4e8f-8c00-9cb1d3ce6a62)

```
{
  "status": "unknown",
  "similarity": 0.08953896909952164,
  "message": "이 음성은 알려진 음성 유형과 일치하지 않습니다."
}
```
🧰 Utilities
remove_silence.py
Uses Silero VAD to remove non-speech regions from audio.
## 📂 Directory Structure
```
├── audio/                          # Temporary folder for uploads
├── dataset/                        # Original grouped audio
├── clean_dataset/                 # Audio after silence removal
├── embedding_labels/              # Saved embeddings & labels
├── final_voice_classifier.pkl     # Trained model
├── *.py                           # All functional scripts
```
📈 Model Details
Embedding Model: speechbrain/spkrec-ecapa-voxceleb

Classifier: RandomForest (200 estimators)

Threshold: Cosine similarity threshold of 0.6

🔒 License
MIT License — use freely with attribution.

🙌 Acknowledgements
* [SpeechBrain](https://speechbrain.readthedocs.io/en/latest/)
* [Silero VAD](https://github.com/snakers4/silero-vad)
* [FastAPI](https://fastapi.tiangolo.com/)



