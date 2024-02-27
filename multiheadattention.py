from lightning import LightningModule
import torch.nn as nn
import math
class MultiHeadAttentionBlock(LightningModule):
    '''
    class to implement multihead attention
    inputs: d_model: dimension of input, h: number of heads, dropout: %dropout
    '''
    def __init__(self, d_model: int, h:int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout= nn.Dropout(dropout)
        assert d_model%h==0, "d_model is not divisible by h"

        self.d_k = d_model//h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
    @staticmethod
    def attention(query, key, value, mask, dropout):
        '''
        class method to calculate attention scores
        we replace the masked values to be near negative infinity
        attention shape= (batch, seq_len, head, dim_k).transpose(1,2) @ (batch, seq_len, head, dim_k) => (batch, head, seq_len, seq_len)
        '''
        d_k = query.shape[-1]
        attention_scores = query @ key.transpose(-2,-1) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask==0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores
    def forward(self,q_i,k_i,v_i,mask):
        '''
        we reshape key, query and values from (batch, seq_len, dim) to be of shape (batch, seq_len, head, dim_k)
        we get the attention scores from class method and reshape it back to original size and apply output weight
        remeber to use .contiguous() in the function as tranpose represents a view doesn't inherently change the storage structure
        '''
        query, key, value = self.w_q(q_i), self.w_k(k_i), self.w_v(v_i)
        query = query.view(query.shape[0], query.shape[1],self.h,  self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)
        x ,self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1,2).contiguous().view(x.shape[0], x.shape[-2], self.h * self.d_k)
        return self.w_o(x)