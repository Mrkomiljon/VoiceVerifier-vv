import os
import numpy as np
import torchaudio
from tqdm import tqdm
from speechbrain.inference import EncoderClassifier

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

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

clean_dataset = r"C:\\Users\\GOOD\\Desktop\\Komil\\Call_center\\clean_dataset"
label_map = {"speaker_1":0, "speaker_2":1, "speaker_3":2}

embeddings, labels = [], []

for speaker_folder in label_map.keys():
    folder_path = os.path.join(clean_dataset, speaker_folder)
    audio_files = [f for f in os.listdir(folder_path) if f.endswith((".wav",".mp3",".flac"))]

    for audio_file in tqdm(audio_files, desc=f"{speaker_folder} embedding"):
        audio_path = os.path.join(folder_path, audio_file)
        emb = get_embedding(audio_path)
        if emb.size > 0:
            embeddings.append(emb)
            labels.append(label_map[speaker_folder])
        else:
            print(f"Embedding olinmadi: {audio_file}")

np.save("final_embeddings.npy", embeddings)
np.save("final_labels.npy", labels)

print("✅ Embeddinglar tayyor!")
