import os
import torchaudio
import joblib
import torch
import numpy as np
from speechbrain.inference import EncoderClassifier
from sklearn.metrics.pairwise import cosine_similarity
from remove_silence import remove_silence, save_audio  

# 사전 학습된 모델 및 임베딩 분류기 로드
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
model = joblib.load("C:\\Users\\GOOD\\Desktop\\Komil\\Call_center\\embedding_labels\\final_voice_classifier.pkl")

# 학습된 임베딩을 불러와 각 클래스에 대한 평균 임베딩 계산
X_train = np.load("C:\\Users\\GOOD\\Desktop\\Komil\\Call_center\\embedding_labels\\X_train.npy")
y_train = np.load("C:\\Users\\GOOD\\Desktop\\Komil\\Call_center\\embedding_labels\\y_train.npy")
class_means = [X_train[y_train == cls].mean(axis=0) for cls in range(3)]

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

# ----- 새로운 오디오를 위한 추론 ----- #

audio_path = "C:\\Users\\GOOD\\Desktop\\Komil\\deepvoicechanger_ys\\audio\\11.wav"

# 정리된 오디오 파일명 생성 (원본 파일명 유지)
file_name = os.path.basename(audio_path)  # 예: '103_Modern.wav'
clean_audio_path = os.path.join("C:\\Users\\GOOD\\Desktop\\Komil\\deepvoicechanger_ys\\audio", f"clean_{file_name}")

# 1단계: 무음 부분 제거
clean_audio, fs = remove_silence(audio_path)

# 🔹 DEBUG: fs 확인
print(f"🟡 디버그: clean_audio 타입: {type(clean_audio)}, 형태: {clean_audio.shape if isinstance(clean_audio, torch.Tensor) else 'N/A'}")
print(f"🟡 디버그: fs 타입: {type(fs)}, 값: {fs}")

# fs가 잘못된 형식이면 16000으로 설정
if not isinstance(fs, int):
    fs = 16000  # 기본 샘플링 속도
if not isinstance(clean_audio, torch.Tensor) or clean_audio.numel() == 0:
    print(f"❌ {file_name} - 오디오가 감지되지 않았거나 빈 임베딩입니다. 파일이 저장되지 않았습니다!")
else:
    # 🔹 `clean_audio`를 올바른 형식으로 변환
    if clean_audio.dtype != torch.float32:
        clean_audio = clean_audio.to(torch.float32)
    
    # 🔹 2D 변환 (torchaudio.save를 위한 조치)
    if clean_audio.dim() == 1:
        clean_audio = clean_audio.unsqueeze(0)

    # 🔹 NumPy 배열로 변환
    clean_audio_np = clean_audio.squeeze().cpu().numpy()

    # 🔹 정리된 오디오 저장
    print(f"🟡 디버그: save_audio 호출 -> 타입: {type(clean_audio_np)}, 형태: {clean_audio_np.shape}, dtype: {clean_audio_np.dtype}")
    
    torchaudio.save(clean_audio_path, torch.tensor(clean_audio_np).unsqueeze(0), fs)
    print(f"✅ 오디오가 정리되었으며 저장되었습니다: {clean_audio_path}")

    # 2단계: 오디오 임베딩 추출
    embedding = get_embedding(clean_audio_path)

    if embedding.size > 0:
        # 3단계: 모델을 통한 유사성 검사 (코사인 유사도)
        similarities = cosine_similarity([embedding], class_means)[0]
        max_similarity = max(similarities)
        predicted_class = np.argmax(similarities)

        print(f"🔹 코사인 유사도: {similarities}")

        # 4단계: 임계값을 통해 최종 결정
        if max_similarity < 0.6:
            print(f"❌ {file_name} - 이 오디오는 기존 클래스와 유사하지 않습니다 (알 수 없는 음성).")
        else:
            print(f"✅ {file_name} - 이 오디오는 {predicted_class+1}번 주요 음성 유형에 속합니다 (유사도: {max_similarity:.2f}).")
    else:
        print(f"❌ {file_name} - 임베딩을 얻을 수 없습니다 (오디오가 너무 짧음).")
