import torch
import torch.nn as nn

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model:int, d_ff: int, dropout:float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout= nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    def forward(self,x):
        return self.linear2(self.dropout(nn.functional.relu(self.linear1(x))))

class ResidualConnection(nn.Module):
    def __init__(self,d_model:int, dropout:float):
        super().__init__()
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x, sublayer):
        x= x + self.dropout(sublayer(self.layernorm(x)))
        return x

class ProjectionLayer(nn.Module):
    def __init__(self, d_model:int, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model,vocab_size)
    def forward(self,x):
        return self.proj(x)