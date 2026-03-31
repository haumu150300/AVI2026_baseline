from torch import nn
from torch.utils.data import Dataset
import os
import pandas as pd 
import numpy as np

TASK1_QS = ["q3", "q4", "q5", "q6"]
TASK2_QS = ["q1", "q2", "q3", "q4", "q5", "q6"]

def load_features_and_labels_task1(base_dir: str, feature_dir: str, label_file: str):
    df = pd.read_csv(label_file)
    data_task1 = {q: [] for q in TASK1_QS}
    labels_task1 = {q: [] for q in TASK1_QS}

    users = set(f.split("_")[0] for f in os.listdir(base_dir))
    for user in users:
        feats = {}
        skip_user = False
        for q in TASK2_QS:
            v_path = os.path.join(feature_dir, "video", f"{user}_{q}.npy")
            a_path = os.path.join(feature_dir, "audio", f"{user}_{q}.npy")
            t_path = os.path.join(feature_dir, "text", f"{user}_{q}.npy")
            if not (os.path.exists(v_path) and os.path.exists(a_path) and os.path.exists(t_path)):
                skip_user = True
                break
            feats[q] = {
                "visual": np.load(v_path),
                "audio": np.load(a_path),
                "text": np.load(t_path)
            }
        if skip_user:
            continue
        # Task1
        for idx, q in enumerate(TASK1_QS):
            label_col = ["H_self","E_self","A_self","C_self"][idx]
            label = df[df["id"]==user][label_col].values[0]
            data_task1[q].append(feats[q])
            labels_task1[q].append(label)
        
    return data_task1, labels_task1


def load_features_and_labels_task2(base_dir: str, feature_dir: str, label_file: str):
    df = pd.read_csv(label_file)
    data_task2 = []
    labels_task2 = []

    users = set(f.split("_")[0] for f in os.listdir(base_dir))
    for user in users:
        feats = {}
        skip_user = False
        for q in TASK2_QS:
            v_path = os.path.join(feature_dir, "video", f"{user}_{q}.npy")
            a_path = os.path.join(feature_dir, "audio", f"{user}_{q}.npy")
            t_path = os.path.join(feature_dir, "text", f"{user}_{q}.npy")
            if not (os.path.exists(v_path) and os.path.exists(a_path) and os.path.exists(t_path)):
                skip_user = True
                break
            feats[q] = {
                "visual": np.load(v_path),
                "audio": np.load(a_path),
                "text": np.load(t_path)
            }
        if skip_user:
            continue
        # Task2
        task2_feat = []
        for q in TASK2_QS:
            f = feats[q]
            task2_feat.extend([f["visual"], f["audio"], f["text"]])
        data_task2.append(task2_feat)
        labels_task2.append(df[df["id"]==user]["g_level"].values[0])
        labels_task2 = [float(x) for x in labels_task2]
    return data_task2, labels_task2

class BaseDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.labels = label
        
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]