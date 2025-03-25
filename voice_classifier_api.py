import os
import torch
import torchaudio
import numpy as np
import joblib
from fastapi import FastAPI, UploadFile, File
from speechbrain.inference import EncoderClassifier
from sklearn.metrics.pairwise import cosine_similarity
from remove_silence import remove_silence

app = FastAPI()

# 🔹 Model va embeddinglar yuklash
print("📦 모델을 로드하는 중...")
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
model = joblib.load("embedding_labels/final_voice_classifier.pkl")
X_train = np.load("embedding_labels/X_train.npy")
y_train = np.load("embedding_labels/y_train.npy")
class_means = [X_train[y_train == cls].mean(axis=0) for cls in range(3)]
print("✅ 모델이 준비되었습니다!")

# 🔹 Foydali funksiya: audioni saqlash
def save_audio(tensor, path, sample_rate):
    if tensor.dtype != torch.float32:
        tensor = tensor.to(torch.float32)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    torchaudio.save(path, tensor, sample_rate)

# 🔹 Embedding olish
def get_embedding(audio_path, segment_duration=10, sample_rate=16000):
    signal, fs = torchaudio.load(audio_path)
    if fs != sample_rate:
        signal = torchaudio.transforms.Resample(fs, sample_rate)(signal)

    total_samples = signal.shape[1]
    segment_samples = segment_duration * sample_rate

    embeddings = []
    for start in range(0, total_samples, segment_samples):
        end = min(start + segment_samples, total_samples)
        segment = signal[:, start:end]
        if segment.shape[1] < int(0.5 * segment_samples):
            continue
        emb = classifier.encode_batch(segment).squeeze().detach().cpu().numpy()
        embeddings.append(emb)

    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.array([])

# 🔹 Asosiy prediksiya funksiyasi
def predict(audio_file, upload_dir="audio"):
    os.makedirs(upload_dir, exist_ok=True)

    filename = os.path.basename(audio_file.filename)
    raw_path = os.path.join(upload_dir, filename)

    with open(raw_path, "wb") as f:
        f.write(audio_file.file.read())

    # 1. remove silence
    clean_audio, fs = remove_silence(raw_path)
    if not isinstance(clean_audio, torch.Tensor) or clean_audio.numel() == 0:
        return {"status": "error", "message": "빈 오디오이거나 음성이 감지되지 않았습니다."}

    clean_path = os.path.join(upload_dir, f"clean_{filename}")
    save_audio(clean_audio, clean_path, fs if isinstance(fs, int) else 16000)

    # 2. extract embedding
    embedding = get_embedding(clean_path)
    if embedding.size == 0:
        return {"status": "error", "message": "임베딩 추출에 실패했습니다."}

    # 3. cosine similarity
    similarities = cosine_similarity([embedding], class_means)[0]
    max_sim = max(similarities)
    pred_class = int(np.argmax(similarities)) + 1

    if max_sim < 0.6:
        return {
            "status": "unknown",
            "similarity": float(max_sim),
            "message": "이 음성은 알려진 음성 유형과 일치하지 않습니다."
        }
    else:
        return {
            "status": "success",
            "predicted_class": pred_class,
            "similarity": float(max_sim),
            "message": f"이 음성은 {pred_class}번 음성 유형과 일치합니다."
        }

# 🔹 FastAPI endpointlari
@app.get("/")
def home():
    return {"message": "음성 분류 API가 실행 중입니다!"}

@app.post("/predict")
async def classify_voice(file: UploadFile = File(...)):
    result = predict(file)
    return result
