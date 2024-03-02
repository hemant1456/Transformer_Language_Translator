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
        self.res_blocks = nn.ModuleList([ResidualConnection(config['d_model'], config['dropout']) for _ in range(3)])
    def forward(self,x, encoder_output, src_mask, tgt_mask):
        x = self.res_blocks[0](x, lambda x: self.attention_block(x,x,x,src_mask))
        x = self.res_blocks[1](x, lambda x: self.cross_attention_block(x,encoder_output,encoder_output,tgt_mask))
        x = self.res_blocks[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers= layers
        self.layernorm = nn.LayerNorm(config["d_model"])
    def forward(self,x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x= layer(x, encoder_output, src_mask, tgt_mask)
        return self.layernorm(x)