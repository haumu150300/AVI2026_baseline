import os
import numpy as np
import pandas as pd

FEATURE_DIR = "/home/orisu/avi2026/dataset/autodl-tmp/train_feature"
BASE_DIR = "/home/orisu/avi2026/dataset/train_data"
LABEL_FILE = "/home/orisu/avi2026/dataset/train_data.csv"
TASK1_QS = ["q3", "q4", "q5", "q6"]
TASK2_QS = ["q1", "q2", "q3", "q4", "q5", "q6"]

def load_features_and_labels():
    df = pd.read_csv(LABEL_FILE)
    data_task1 = {q: [] for q in TASK1_QS}
    labels_task1 = {q: [] for q in TASK1_QS}
    data_task2 = []
    labels_task2 = []

    users = set(f.split("_")[0] for f in os.listdir(BASE_DIR))
    for user in users:
        feats = {}
        skip_user = False
        for q in TASK2_QS:
            v_path = os.path.join(FEATURE_DIR, "video", f"{user}_{q}.npy")
            a_path = os.path.join(FEATURE_DIR, "audio", f"{user}_{q}.npy")
            t_path = os.path.join(FEATURE_DIR, "text", f"{user}_{q}.npy")
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
        
        # Task2
        task2_feat = []
        for q in TASK2_QS:
            f = feats[q]
            task2_feat.extend([f["visual"], f["audio"], f["text"]])
        data_task2.append(task2_feat)
        labels_task2.append(df[df["id"]==user]["g_level"].values[0])
        labels_task2 = [float(x) for x in labels_task2]
    return data_task1, labels_task1, data_task2, labels_task2

 