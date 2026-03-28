import os
import torch
import librosa
import subprocess
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np

# ====================== 配置 ======================
BASE_DIR = "/home/orisu/avi2026/dataset/train_data"       # 原始视频文件夹
SAVE_DIR = "/home/orisu/avi2026/dataset/autodl-tmp"            # 保存音频和文本的根文件夹
AUDIO_DIR = os.path.join(SAVE_DIR, "train_audio")
TEXT_DIR = os.path.join(SAVE_DIR, "train_text")
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 加载 Whisper 模型
print("Loading Whisper model...")
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(device)
print("Whisper model loaded.")

# ====================== 工具函数 ======================
def extract_audio_from_video(video_path, audio_path):
    """使用 ffmpeg 提取视频音频并保存为 wav"""
    if os.path.exists(audio_path):
        return
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-ar", "16000",
        "-ac", "1",
        audio_path
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def split_audio(audio, sr, chunk_sec=30):
    chunk_size = sr * chunk_sec
    return [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size)]

def extract_text_from_audio(audio_path, text_path):
    audio, sr = librosa.load(audio_path, sr=16000)  # MUST be 16kHz
    chunks = split_audio(audio, 16000)

    full_text = []
    for chunk in chunks:
        inputs = whisper_processor(chunk, sampling_rate=16000, return_tensors="pt").to(device)

        with torch.no_grad():
            ids = whisper_model.generate(inputs["input_features"])

        text = whisper_processor.batch_decode(ids, skip_special_tokens=True)[0]
        full_text.append(text)

    final_text = " ".join(full_text)
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(final_text.strip())
        
# ====================== 批量处理视频 ======================
videos = [f for f in os.listdir(BASE_DIR) if f.endswith(".mp4")]
print(f"Found {len(videos)} videos.")

for video_file in videos:
    user_q = "_".join(video_file.split("_")[:2])
    video_path = os.path.join(BASE_DIR, video_file)
    
    # 音频保存路径
    audio_path = os.path.join(AUDIO_DIR, f"{user_q}.wav")
    text_path = os.path.join(TEXT_DIR, f"{user_q}.txt")
    
    
    try:
        print(f"\nProcessing {video_file} ...")
        # 提取音频
        extract_audio_from_video(video_path, audio_path)
        extract_text_from_audio(audio_path, text_path)
        print(f"  ▶ Audio saved to {audio_path}")
        
    except Exception as e:
        print(f"❌ Failed processing {video_file}: {e}")

print("\n✅ All videos processed. Audio saved in:")
print(f"  Audio folder: {AUDIO_DIR}")
