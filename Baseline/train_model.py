import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data_utils import load_features_and_labels
import torch.nn.functional as F
import random

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多 GPU 情况

    # 保证 cuDNN 可复现（会略微降低速度，但 baseline 完全可以接受）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch >= 1.8
    torch.use_deterministic_algorithms(True)

# 设置全局随机种子
set_seed(42)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
TASK1_QS = ["q3", "q4", "q5", "q6"]
TASK2_QS = ["q1", "q2", "q3", "q4", "q5", "q6"]

# ------------------------------
# Personality Regressor (Expert)
# ------------------------------
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)
    
class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        return attn_output
    
class PersonalityRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], num_heads=1):
        super().__init__()
        # self.fusion = nn.Linear(input_dim, hidden_dims[0])
         
        self.fusion = Residual(PreNorm(input_dim, Attention(input_dim, 8)))
        self.hidden_layers = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dims[1], 1) for _ in range(num_heads)])

    def add_noise(self, x, std=0.01):
        noise = torch.randn_like(x) * std
        return x + noise
    

    def forward(self, x):
        x = self.add_noise(x, std=0.001)
        x = self.fusion(x)
        x = self.hidden_layers(x)
        outputs = [head(x) for head in self.heads]
        return torch.mean(torch.stack(outputs, dim=0), dim=0)  # (batch, 1)


class PersonalityRegressorDefault(nn.Module):
    def __init__(self, visual_dim=768, audio_dim=1024, text_dim=768, hidden_dims=[256,128], num_heads=5):
        super().__init__()
        self.fusion = nn.Linear(visual_dim + audio_dim + text_dim, hidden_dims[0])
        self.hidden_layers = nn.Sequential(
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(), nn.Dropout(0.2)
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dims[1], 1) for _ in range(num_heads)])

    def forward(self, v,a,t):
        x = torch.cat([v,a,t], dim=1)
        x = self.fusion(x)
        x = self.hidden_layers(x)
        outputs = [head(x) for head in self.heads]
        return torch.mean(torch.stack(outputs, dim=0), dim=0)
# ------------------------------
# Mixture of Experts Model
# ------------------------------
class PersonalityMoe(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Learnable mixture weights (optional softmax for stability)
        # self.weights = nn.Parameter(torch.tensor([0.2, 0.2, 0.6], dtype=torch.float32))  

        self.expert_v = PersonalityRegressor(512).to(device)
        self.expert_a = PersonalityRegressor(512).to(device)
        self.expert_t = PersonalityRegressor(768).to(device)

    def forward(self, v, a, t):
        out_v = self.expert_v(v)  # (batch, 1)
        out_a = self.expert_a(a)
        out_t = self.expert_t(t)

        # Normalize weights to sum=1
        # w = F.softmax(self.weights, dim=0)
        # output = w[0]*out_v + w[1]*out_a + w[2]*out_t
        # return output.squeeze(-1)  # (batch,)
        return (out_v + out_a + out_t) / 3.0  # 简单平均

    
class CognitiveClassifier(nn.Module):
    def __init__(self, visual_dim=768, audio_dim=1024, text_dim=768, hidden_dims=[512,256], num_heads=3):
        super().__init__()
        self.single_fusion = nn.Linear(visual_dim+audio_dim+text_dim, hidden_dims[0])
        self.global_fusion = nn.Linear(hidden_dims[0]*6, hidden_dims[1])
        self.heads = nn.ModuleList([nn.Sequential(nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_dims[1],3)) for _ in range(num_heads)])

    def forward(self, *inputs):
        q_feats=[]
        for i in range(0,len(inputs),3):
            x = torch.cat(inputs[i:i+3], dim=1)
            q_feats.append(self.single_fusion(x))
        global_feat = torch.cat(q_feats, dim=1)
        global_feat = self.global_fusion(global_feat)
        outputs = [head(global_feat) for head in self.heads]
        return torch.mean(torch.stack(outputs, dim=0), dim=0)

# ====================== 主流程 ======================
if __name__=="__main__":
    # 确保保存模型的目录存在
    os.makedirs("./trained_models", exist_ok=True)

    data_task1, labels_task1, data_task2, labels_task2 = load_features_and_labels()

    # Task1
    print("\n=== Task1: 人格回归训练 ===")
    
    task1_models = {}
    for idx, q in enumerate(TASK1_QS):
        X = data_task1[q]
        y = labels_task1[q]
        model = PersonalityRegressorDefault()
        model.to(device)
        
        # print architecture and parameter is trainable
        # print(f"\nModel architecture for {q}:")
        # for name, param in model.named_parameters():
        #     print(f"{name}: requires_grad={param.requires_grad}, shape={param.shape}")
        # exit()
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        Xv = torch.tensor(np.stack([f["visual"] for f in X]), dtype=torch.float32).to(device)
        Xa = torch.tensor(np.stack([f["audio"]  for f in X]), dtype=torch.float32).to(device)
        Xt = torch.tensor(np.stack([f["text"]   for f in X]), dtype=torch.float32).to(device)
        Y  = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)

        model.train()
        for epoch in range(500):
            optimizer.zero_grad()
            output = model(Xv, Xa, Xt)
            loss = criterion(output,Y)
            loss.backward()
            optimizer.step()
            if epoch%10==0:
                print(f"{q} Epoch {epoch} | MSE Loss: {loss.item():.6f}")
        task1_models[q] = model
        torch.save(model.state_dict(), f"./trained_models/task1_{q}.pth")

    # Task2
    print("\n=== Task2: 认知分类训练 ===")
    print('labels_task2: ', len(labels_task2))
    print('data_task2: ', set(labels_task2))
    level_map = {"low":0,"normal":1,"high":2}
    # Y2 = [level_map[label] for label in labels_task2]
    Y2  = labels_task2
    
    # 构建 tensor（N 用户 × 18 个特征）
    # 每个用户有 6 个问题 × 3 个模态 = 18 个 feature
    X2_tensor_list = []
    for feats in data_task2:
        # feats 是长度 18 的 list，分别是每个 q 的 [v,a,t]
        X_user = [torch.tensor(f, dtype=torch.float32) for f in feats]  # list of tensors
        X2_tensor_list.append(X_user)
    
    # 将 list 转成 Numpy stack 再转 tensor
    # 先把每个用户的 18 个 feature concat 成 [18, dim] 再 stack 成 [N, 18, dim]
    batch_size = len(X2_tensor_list)  # 全量 batch
    X2_batch = []
    for i in range(18):
        X2_batch.append(torch.stack([X2_tensor_list[j][i] for j in range(batch_size)], dim=0).to(device))
    
    Y2_tensor = torch.tensor(Y2, dtype=torch.long).to(device)
    Y2_tensor = Y2_tensor - 1 # force labels to be 0,1,2
    print('Y2_tensor: ', Y2_tensor.shape)
    
    model2 = CognitiveClassifier().to(device)
    criterion2 = nn.CrossEntropyLoss()
    optimizer2 = optim.Adam(model2.parameters(), lr=1e-4)
    
    num_epochs = 500
    for epoch in range(num_epochs):
        optimizer2.zero_grad()
        output = model2(*X2_batch)  # [N, 3]
        loss = criterion2(output, Y2_tensor)
        loss.backward()
        optimizer2.step()
    
        if epoch % 10 == 0:
            pred = torch.argmax(output, dim=1)
            acc = (pred == Y2_tensor).float().mean().item()
            print(f"Epoch {epoch} | CE Loss: {loss.item():.6f} | Acc: {acc:.4f}")
    
    torch.save(model2.state_dict(), "./trained_models/task2_cognitive.pth")
    print("\n🎉 所有任务运行完成！")

