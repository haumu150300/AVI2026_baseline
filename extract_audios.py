import os
import torch
import librosa
import subprocess
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ====================== 配置 ======================
BASE_DIR = "/root/autodl-tmp/val_data"       # 原始视频文件夹
SAVE_DIR = "/root/autodl-tmp"            # 保存音频和文本的根文件夹
AUDIO_DIR = os.path.join(SAVE_DIR, "val_audio")
TEXT_DIR = os.path.join(SAVE_DIR, "val_text")
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

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
    cmd = f"ffmpeg -y -i {video_path} -ar 16000 -ac 1 {audio_path}"
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ====================== 批量处理视频 ======================
videos = [f for f in os.listdir(BASE_DIR) if f.endswith(".mp4")]
print(f"Found {len(videos)} videos.")

for video_file in videos:
    user_q = "_".join(video_file.split("_")[:2])
    video_path = os.path.join(BASE_DIR, video_file)
    
    # 音频保存路径
    audio_path = os.path.join(AUDIO_DIR, f"{user_q}.wav")
    
    try:
        print(f"\nProcessing {video_file} ...")
        # 提取音频
        extract_audio_from_video(video_path, audio_path)
        print(f"  ▶ Audio saved to {audio_path}")
        
    except Exception as e:
        print(f"❌ Failed processing {video_file}: {e}")

print("\n✅ All videos processed. Audio saved in:")
print(f"  Audio folder: {AUDIO_DIR}")
