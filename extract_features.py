import os
import cv2
import torch
import numpy as np
import librosa
import warnings
from transformers import CLIPProcessor, CLIPModel, WhisperProcessor, WhisperModel, RobertaTokenizer, RobertaModel

warnings.filterwarnings('ignore')

# 基础配置
device = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = "/root/autodl-tmp/val_data"
BASE_VIDEO_DIR = BASE_DIR
BASE_AUDIO_DIR = "/root/autodl-tmp/val_audio"
BASE_TEXT_DIR  = "/root/autodl-tmp/val_text"
FEATURE_DIR = "/root/autodl-tmp/val_feature"
os.makedirs(os.path.join(FEATURE_DIR, "video"), exist_ok=True)
os.makedirs(os.path.join(FEATURE_DIR, "audio"), exist_ok=True)
os.makedirs(os.path.join(FEATURE_DIR, "text"), exist_ok=True)
TASK2_QS = ["q1", "q2", "q3", "q4", "q5", "q6"]

# 模型加载
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
whisper_model = WhisperModel.from_pretrained("openai/whisper-base").to(device)
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta_model = RobertaModel.from_pretrained("roberta-base").to(device)

# ====================== 工具函数 ======================
def extract_keyframes(video_path, fps=1):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(int(frame_rate * fps), 1)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        count += 1
    cap.release()
    if not frames:
        frames = [np.zeros((224,224,3), dtype=np.uint8)]
    return frames

def get_video_path(user_id, q_id):
    for file in os.listdir(BASE_VIDEO_DIR):
        if file.startswith(f"{user_id}_{q_id}_") and file.endswith(".mp4"):
            return os.path.join(BASE_VIDEO_DIR, file)
    return None

def get_audio_path(user_id, q_id):
    path = os.path.join(BASE_AUDIO_DIR, f"{user_id}_{q_id}.wav")
    return path if os.path.exists(path) else None

def get_text_path(user_id, q_id):
    path = os.path.join(BASE_TEXT_DIR, f"{user_id}_{q_id}.txt")
    return path if os.path.exists(path) else None

# ====================== 特征提取函数 ======================
def extract_visual_feature(video_path):
    frames = extract_keyframes(video_path)
    inputs = clip_processor(images=frames, return_tensors="pt").to(device)
    with torch.no_grad():
        feat = clip_model.get_image_features(**inputs)
    return torch.max(feat, dim=0).values.cpu().numpy()

def extract_audio_feature(audio_path):
    waveform, sr = librosa.load(audio_path, sr=16000)
    inputs = whisper_processor(waveform, sampling_rate=16000, return_tensors="pt").to(device)
    with torch.no_grad():
        feat = whisper_model.encoder(**inputs).last_hidden_state
    return torch.max(feat, dim=1).values.squeeze().cpu().numpy()

def extract_text_feature(text_path):
    try:
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
    except UnicodeDecodeError:
        with open(text_path, "r", encoding="gbk") as f:
            text = f.read().strip()
    inputs = roberta_tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=512, padding="max_length").to(device)
    with torch.no_grad():
        feat = roberta_model(**inputs).last_hidden_state
    return torch.max(feat, dim=1).values.squeeze().cpu().numpy()

def batch_extract_features():
    users = set(f.split("_")[0] for f in os.listdir(BASE_DIR))
    for user in users:
        for q in TASK2_QS:
            video_path = get_video_path(user, q)
            audio_path = get_audio_path(user, q)
            text_path  = get_text_path(user, q)
            if video_path and audio_path and text_path:
                v_feat = extract_visual_feature(video_path)
                a_feat = extract_audio_feature(audio_path)
                t_feat = extract_text_feature(text_path)
                np.save(os.path.join(FEATURE_DIR, "video", f"{user}_{q}.npy"), v_feat)
                np.save(os.path.join(FEATURE_DIR, "audio", f"{user}_{q}.npy"), a_feat)
                np.save(os.path.join(FEATURE_DIR, "text", f"{user}_{q}.npy"), t_feat)
                print(f"✅ 提取并保存 {user}_{q} 的特征")


if __name__ == "__main__":
    batch_extract_features()
