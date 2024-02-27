import torch
import torch.nn as nn
from lightning import LightningModule

class FeedForwardBlock(LightningModule):
    def __init__(self, d_model:int, d_ff: int, dropout:float):
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout= nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    def forward(self,x):
        return self.linear2(self.dropout(nn.functional.relu(self.linear1(x))))

class ResidualConnection(LightningModule):
    def __init__(self,d_model:int, dropout:float):
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x, sublayer):
        x= x + self.dropout(sublayer(self.norm(x)))
        return x

class ProjectionLayer(LightningModule):
    def __init__(self, d_model:int, vocab_size):
        self.proj = nn.Linear(d_model,vocab_size)
    def forward(self,x):
        return nn.functional.log_softmax(self.proj(x), dim=-1)