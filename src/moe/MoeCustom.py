import torch
import torch.nn as nn
from einops import rearrange

# Question: visual - shape: (512,)
# Question: audio - shape: (512,)
# Question: text - shape: (768,)

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
    
class MoeCustom(nn.Module):
    def __init__(self, visual_dim=512, audio_dim=512, text_dim=768, hidden_dims=[256,128], num_heads=3):
        super().__init__()
        
        self.visual_block = Residual(PreNorm(visual_dim, Attention(visual_dim, 8)))
        self.audio_block = Residual(PreNorm(audio_dim, Attention(audio_dim, 8)))
        self.text_block = Residual(PreNorm(text_dim, Attention(text_dim, 8)))
        self.weights = torch.tensor([0.2, 0.2, 0.6])  # visual, audio, text weights
        
        self.hidden_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(), nn.Dropout(0.2)
        ) for input_dim in [visual_dim, audio_dim, text_dim]])
        
        self.heads = nn.ModuleList([nn.Linear(hidden_dims[1], 1) for _ in range(num_heads)])

    def forward(self, v,a,t):
        v = self.visual_block(v).mean(dim=1)  # (b, 512)
        a = self.audio_block(a).mean(dim=1)  # (b, 512)
        t = self.text_block(t).mean(dim=1)   # (b, 768)
        # print("After attention - visual:", v.shape, "audio:", a.shape, "text:", t.shape)
        
        v = self.hidden_layers[0](v)  # (b, 128)
        a = self.hidden_layers[1](a)  # (b, 128)
        t = self.hidden_layers[2](t)  # (b, 128)
        # print("After hidden layers - visual:", v.shape, "audio:", a.shape, "text:", t.shape)

        v_outputs = self.heads[0](v)
        a_outputs = self.heads[1](a)
        t_outputs = self.heads[2](t)
        return sum(v_outputs) + sum(a_outputs) + sum(t_outputs)
    
# model = MoeCustom()
# x =torch.randn(1, 2, 512)  # visual
# a =torch.randn(1, 2, 512)  # audio
# t =torch.randn(1, 2, 768)  # text
# out = model(x, a, t)
# print(out)