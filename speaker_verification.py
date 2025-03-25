import torch
import torchaudio
import os
import numpy as np
import pickle
from pyannote.audio import Model
from sklearn.metrics.pairwise import cosine_similarity

# ====== 1. Modelni yuklash ======
device = "cuda" if torch.cuda.is_available() else "cpu"
spk_model = Model.from_pretrained("pyannote/embedding").to(device)
print("✅ `pyannote/embedding` modeli yuklandi!")

# ====== 2. Audio Faylni Yuklash va Embedding Hosil Qilish ======
def load_audio(audio_path, target_sample_rate=16000):
    """Audio faylni yuklash va mono formatga o'tkazish"""
    waveform, sample_rate = torchaudio.load(audio_path)
    
    if waveform.shape[0] > 1:
        print(f"⚠️ Stereo audio aniqlandi. Mono formatga o'tkazilmoqda: {audio_path}")
        waveform = waveform.mean(dim=0, keepdim=True)
    
    if sample_rate != target_sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)(waveform)
    
    return waveform

def get_pyannote_embedding(audio_path):
    """Pyannote speaker embedding olish va saqlash"""
    embedding_cache = f"{audio_path}.embedding.pkl"
    
    # Agar embedding avval yaratilgan bo'lsa, uni yuklaymiz
    if os.path.exists(embedding_cache):
        with open(embedding_cache, "rb") as f:
            print(f"📂 '{audio_path}' uchun oldindan yaratilgan embedding yuklanmoqda...")
            return pickle.load(f)
    
    # Yangi embedding yaratish
    print(f"🔄 '{audio_path}' uchun yangi embedding hosil qilinmoqda...")
    waveform = load_audio(audio_path).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = spk_model(waveform).squeeze().cpu().numpy()
    
    # Saqlash
    with open(embedding_cache, "wb") as f:
        pickle.dump(embedding, f)
    
    return embedding

# ====== 3. Speaker Identification ======
def identify_speaker(test_audio_path, reference_paths):
    """Ovoz kimga tegishli ekanligini aniqlash"""
    reference_embeddings = {name: get_pyannote_embedding(path) for name, path in reference_paths.items()}
    test_embedding = get_pyannote_embedding(test_audio_path)
    
    similarities = {name: cosine_similarity([test_embedding], [emb])[0][0] for name, emb in reference_embeddings.items()}
    best_match = max(similarities, key=similarities.get)

    # Natijalarni log faylga yozamiz
    log_file = "speaker_verification_results.log"
    with open(log_file, "a") as f:
        f.write(f"\nTest Audio: {test_audio_path}\n")
        for name, score in similarities.items():
            f.write(f"{name}: {score:.3f}\n")
        f.write(f"Final Match: {best_match}\n{'='*40}\n")

    # Konsolga natijalarni chiqaramiz
    print("\n🔍 Ovoz o'xshashligi natijalari:")
    for name, score in similarities.items():
        print(f"✅ {name}: {score:.3f}")
    
    print(f"\n🎤 Bu ovoz eng ko‘p **{best_match}** ga o‘xshash!")

# ====== 4. Test ======
if __name__ == "__main__":
    reference_paths = {
        "Speaker 1": "C:\\Users\\GOOD\\Desktop\\Komil\\Call_center\\client_voices\\Female_LoFi_origin_voice.wav",
        "Speaker 2": "C:\\Users\\GOOD\\Desktop\\Komil\\Call_center\\client_voices\\Male_Emo_Pop_origin_voice.wav",
        "Speaker 3": "C:\\Users\\GOOD\\Desktop\\Komil\\Call_center\\client_voices\\Modern_Male_HipHop_original_voice.wav"
    }
    test_audio_path = "C:\\Users\\GOOD\\Desktop\\Komil\\deepvoicechanger_ys\\audio\\103_Modern.wav"
    
    identify_speaker(test_audio_path, reference_paths)
