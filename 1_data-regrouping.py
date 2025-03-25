import os
import shutil
from tqdm import tqdm

# Original fayllar joylashgan papka
source_dir = r"C:\Users\GOOD\Desktop\Komil\deepvoicechanger_ys\data\fake"

# Yangi saqlanadigan papka
target_dir = r"C:\Users\GOOD\Desktop\Komil\Call_center\dataset"

# Kalit so'z va speaker papka nomlari
keywords = {
    "modern": "speaker_1",
    "lofi": "speaker_2",
    "emo": "speaker_3"
}

# Agar target papka yo‘q bo‘lsa yaratish
for speaker_folder in keywords.values():
    os.makedirs(os.path.join(target_dir, speaker_folder), exist_ok=True)

# Audio fayllarni nomiga qarab ajratish
audio_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.wav', '.mp3', '.flac'))]

for audio_file in tqdm(audio_files, desc="Audiolar ko'chirilmoqda"):
    lower_audio_name = audio_file.lower()

    copied = False
    for key, speaker_folder in keywords.items():
        if key in lower_audio_name:
            source_file = os.path.join(source_dir, audio_file)
            target_file = os.path.join(target_dir, speaker_folder, audio_file)
            shutil.copy(source_file, target_file)
            copied = True
            break  # Kalit so'z topildi, boshqa tekshirish shart emas.

    if not copied:
        print(f"❌ {audio_file} hech qaysi kalit so'zga mos kelmadi va ko'chirilmadi.")

print("✅ Barcha audiolar muvaffaqiyatli ko'chirildi!")
