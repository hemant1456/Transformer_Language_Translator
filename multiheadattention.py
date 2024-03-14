from lightning import LightningModule
import torch.nn as nn
import math
from configuration import get_config

config = get_config()


class MultiHeadAttentionBlock(nn.Module):
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

        input shapes for key, query and values is (batch, heads, seq_len, d_model//heads)
        input shape for mask is (seq_len, seq_len)
        
        we replace the masked values to be near negative infinity to get 0 value in softmax
        attention shape= (batch, head, seq_len, dim_k).transpose(1,2) @ (batch, head, seq_len, dim_k) => (batch, head, seq_len, seq_len)
        '''
        assert value.size(1)==key.size(1)==query.size(1)== config["heads"], "incorrect dimensions for either of key query of value"
        assert value.size(-1)==key.size(-1)==query.size(-1)== config["d_model"]//config["heads"], "incorrect dimensions for either of key query of value"
        
        d_k = query.shape[-1]
        attention_scores = query @ key.transpose(-2,-1) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask, -1e9)
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
        x = x.transpose(1,2).contiguous() #interchange head and seq_len dimension
        x = x.view(x.shape[0], x.shape[1], self.h * self.d_k)
        return self.w_o(x)