import os
import torch
import numpy as np
import pandas as pd
from train_model import PersonalityRegressorDefault, CognitiveClassifier, TASK1_QS, TASK2_QS

device = "cuda" if torch.cuda.is_available() else "cpu"

FEATURE_DIR = "/home/orisu/avi2026/dataset/autodl-tmp1/val_feature"  # 验证集特征目录
VAL_CSV = "/home/orisu/avi2026/dataset/val_data.csv"

# ====================== 加载验证集数据 ======================
def load_val_features_and_labels():
    df = pd.read_csv(VAL_CSV)
    
    data_task1 = {q: [] for q in TASK1_QS}
    labels_task1 = {q: [] for q in TASK1_QS}
    data_task2 = []
    labels_task2 = []

    level_map = {"low":0, "normal":1, "high":2}

    for _, row in df.iterrows():
        user_id = str(row['id'])
        feats = {}
        skip_user = False
        for q in TASK2_QS:
            v_path = os.path.join(FEATURE_DIR, "video", f"{user_id}_{q}.npy")
            a_path = os.path.join(FEATURE_DIR, "audio", f"{user_id}_{q}.npy")
            t_path = os.path.join(FEATURE_DIR, "text",  f"{user_id}_{q}.npy")
            if not (os.path.exists(v_path) and os.path.exists(a_path) and os.path.exists(t_path)):
                skip_user = True
                break
            feats[q] = {
                "visual": np.load(v_path).mean(axis=0),
                "audio": np.load(a_path).squeeze(),
                "text": np.load(t_path)
            }
        if skip_user:
            continue

        # Task1
        # 假设列名和训练集一致
        label_cols = ["H_self","E_self","A_self","C_self"]
        for idx, q in enumerate(TASK1_QS):
            labels_task1[q].append(row[label_cols[idx]])
            data_task1[q].append(feats[q])

        # Task2
        task2_feat = []
        for q in TASK2_QS:
            f = feats[q]
            task2_feat.extend([f["visual"], f["audio"], f["text"]])
        data_task2.append(task2_feat)
        # labels_task2.append(level_map[row['g_level']])  # 保持你原来的列名
        labels_task2.append(row['g_level'])  # 保持你原来的列名
    return data_task1, labels_task1, data_task2, labels_task2


# ====================== 验证 Task1 ======================
def evaluate_task1(task1_models, data_task1, labels_task1):
    from sklearn.metrics import mean_squared_error
    print("\n=== Task1 验证集 ===")
    avg_scores = 0
    for q in TASK1_QS:
        X = data_task1[q]
        y_true = labels_task1[q]
        if not X:
            continue
        Xv = torch.tensor(np.stack([f["visual"] for f in X]), dtype=torch.float32).to(device)
        Xa = torch.tensor(np.stack([f["audio"]  for f in X]), dtype=torch.float32).to(device)
        Xt = torch.tensor(np.stack([f["text"]   for f in X]), dtype=torch.float32).to(device)

        model = task1_models[q]
        model.eval()
        with torch.no_grad():
            y_pred = model(Xv,Xa,Xt).cpu().numpy()
        mse = mean_squared_error(y_true, y_pred)
        print(f"{q} MSE: {mse:.6f}")
        avg_scores += mse
    avg_scores /= len(TASK1_QS)
    print(f"Average MSE across Task1: {avg_scores:.6f}")

# ====================== 验证 Task2 ======================
def evaluate_task2(model2, data_task2, labels_task2):
    print("\n=== Task2 验证集 ===")
    correct = 0
    total = len(data_task2)
    model2.eval()
    Y_true = []
    Y_pred = []
    with torch.no_grad():
        for i, X_user in enumerate(data_task2):
            X_user_tensors = [torch.tensor(f, dtype=torch.float32).unsqueeze(0).to(device) for f in X_user]
            output = model2(*X_user_tensors)
            pred = torch.argmax(output, dim=1).item()
            Y_pred.append(pred)
            Y_true.append(labels_task2[i])
            correct += (pred == labels_task2[i])
    acc = correct / total if total>0 else 0
    print(f"Accuracy: {acc:.4f}")
    return Y_true, Y_pred

# ====================== 主流程 ======================
if __name__=="__main__":
    # 加载模型
    task1_models = {}
    for q in TASK1_QS:
        model = PersonalityRegressorDefault()
        model.load_state_dict(torch.load(f"./trained_models/task1_{q}.pth", map_location=device))
        model.to(device)
        task1_models[q] = model

    model2 = CognitiveClassifier().to(device)
    model2.load_state_dict(torch.load("./trained_models/task2_cognitive.pth", map_location=device))

    # 加载验证集
    data_task1, labels_task1, data_task2, labels_task2 = load_val_features_and_labels()

    # 评估
    evaluate_task1(task1_models, data_task1, labels_task1)
    evaluate_task2(model2, data_task2, labels_task2)
