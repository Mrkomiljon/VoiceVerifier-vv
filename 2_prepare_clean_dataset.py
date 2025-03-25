import os
from tqdm import tqdm
from remove_silence import remove_silence, save_clean_audio

original_dataset = r"C:\Users\GOOD\Desktop\Komil\Call_center\dataset"
clean_dataset = r"C:\Users\GOOD\Desktop\Komil\Call_center\clean_dataset"

for speaker_folder in ["speaker_1", "speaker_2", "speaker_3"]:
    source_folder = os.path.join(original_dataset, speaker_folder)
    target_folder = os.path.join(clean_dataset, speaker_folder)
    os.makedirs(target_folder, exist_ok=True)

    audio_files = [f for f in os.listdir(source_folder) if f.endswith((".wav",".mp3",".flac"))]

    for audio_file in tqdm(audio_files, desc=f"{speaker_folder} cleaning"):
        audio_path = os.path.join(source_folder, audio_file)
        
        clean_audio, fs = remove_silence(audio_path, threshold=0.3)

        if clean_audio.numel() > 0:
            target_audio_path = os.path.join(target_folder, audio_file)
            save_clean_audio(clean_audio, fs, target_audio_path)
        else:
            print(f"{audio_file} ovozli qismi aniqlanmadi.")

print("✅ Ovozsiz joylar Silero-VAD yordamida olib tashlandi!")
