import torch.nn as nn
from model_component import ResidualConnection

from configuration import get_config
config = get_config()

class EncoderBlock(nn.Module):
    def __init__(self, attention_block, feed_forward_block):
        super().__init__()
        self.attention_block = attention_block
        self.feed_forward_block = feed_forward_block
        self.res_blocks = [ResidualConnection(config['d_model'], config['dropout']) for _ in range(2)]
    def forward(self,x, src_mask):
        x = self.res_blocks[0](x, lambda x: self.attention_block(x,x,x,src_mask))
        x = self.res_blocks[1](self.feed_forward_block)

class Encoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers= layers
        self.layernorm = nn.LayerNorm(config["d_model"])
    def forward(self,x, mask):
        for layer in self.layers:
            x= layer(x, mask)
        return self.layernorm(x)

