import torch
import torchaudio

# Silero VAD modelini yuklash
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

def remove_silence(audio_path, sample_rate=16000, threshold=0.3):
    # audiolarni yuklash va kerak bo'lsa resample qilish
    wav = read_audio(audio_path, sampling_rate=sample_rate)
    
    # ovozli qismlarni aniqlash
    speech_timestamps = get_speech_timestamps(wav, model, threshold=threshold, sampling_rate=sample_rate)
    
    # ovozli qismlarini birlashtirish
    if speech_timestamps:
        speech_audio = collect_chunks(speech_timestamps, wav)
        speech_audio = torch.tensor(speech_audio).unsqueeze(0)
    else:
        speech_audio = torch.tensor([])

    return speech_audio, sample_rate

def save_clean_audio(signal, fs, save_path):
    torchaudio.save(save_path, signal, fs)
