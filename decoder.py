import torch.nn as nn
from model_component import ResidualConnection

from configuration import get_config
config = get_config()

class DecoderBlock(nn.Module):
    def __init__(self, attention_block, cross_attention_block, feed_forward_block):
        super().__init__()
        self.attention_block = attention_block
        self.cross_attention_block= cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.res_blocks = [ResidualConnection(config['d_model'], nn.Dropout(config['dropout'])) for _ in range(3)]
    def forward(self,x, encoder_output, src_mask, tgt_mask):
        x = self.res_blocks[0](x, lambda x: self.attention_block(x,x,x,src_mask))
        x = self.res_blocks[1](x, lambda x: self.cross_attention_block(x,encoder_output,encoder_output,tgt_mask))
        x= self.res_blocks[2](self.feed_forward_block)

class Decoder(nn.Module):
    def __init__(self, layers):
        self.layers= layers
        self.layernorm = nn.LayerNorm(config["d_model"])
        super().__init__()
    def forward(self,x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x= layer(x, encoder_output, src_mask, tgt_mask)
        return self.layernorm(x)