import torch.nn as nn
from model_component import FeedForwardBlock, ResidualConnection
from multiheadattention import MultiHeadAttentionBlock

from configuration import get_config
config = get_config()

class EncoderBlock(nn.Module):
    def __init__(self, x):
        super().__init__()
        self.attention_block = MultiHeadAttentionBlock(config['d_model'], config['heads'], config['dropout'])
        self.feed_forward_block = FeedForwardBlock(config['d_model'], config['d_ff'], config['dropout'])
        self.res_block1 = ResidualConnection(config['d_model'], nn.Dropout(config['dropout']))
    def forward(self,x, src_mask):
        x = self.res_block1(x, lambda x: self.attention_block(x,x,x,src_mask))

class Encoder(nn.Module):
    def __init__(self, layers):
        self.layers= layers
        self.layernorm = nn.LayerNorm(config["d_model"])
        super().__init__()
    def forward(self,x, mask):
        for layer in self.layers:
            x= layer(x, mask)
        return self.layernorm(x)

