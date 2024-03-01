from lightning import LightningModule
from embedding import InputEmbedding, PositionalEmbedding
from configuration import get_config
from encoder import Encoder, EncoderBlock
from multiheadattention import MultiHeadAttentionBlock
from model_component import FeedForwardBlock, ProjectionLayer
from decoder import Decoder, DecoderBlock
import torch.nn as nn

config = get_config()

class Transformer(LightningModule):
    def __init__(self, src_embed, tgt_embed, src_pos, tgt_pos, encoder, decoder, project):
        super().__init__()
        self.src_embed= src_embed
        self.tgt_embed= tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.encoder = encoder
        self.decoder= decoder
        self.project= project
    def training_step(self, batch, batch_idx):
        enc_inputs, dec_inputs, src_mask, tgt_mask, labels, src_texts, tgt_texts = batch 
        encoder_output = self.encode(x, src_mask)
        x = self.decode(dec_inputs, encoder_output, src_mask, tgt_mask)
        x = self.project(x)
    def encode(self,x, src_mask):
        x = self.src_embed(x)
        x = self.tgt_pos(x)
        x = self.encoder(x, src_mask)
        return x
    def decode(self, x, encoder_output, src_mask, tgt_mask):
        x= self.tgt_embed(x)
        x = self.tgt_pos(x)
        x = self.decode(x, encoder_output, src_mask, tgt_mask)
        return x
    def decode(self,x):
        x = self.project(x)
        return x
    
def build_transformer(tokenizer_src, tokenizer_tgt):
    src_embed = InputEmbedding(config["d_model"], tokenizer_src.get_vocab_size())
    tgt_embed = InputEmbedding(config["d_model"], tokenizer_tgt.get_vocab_size())
    src_pos = PositionalEmbedding(config["d_model"], config["seq_len"])
    tgt_pos = PositionalEmbedding(config["d_model"], config["seq_len"])

    encoder_blocks = []
    for _ in range(config["num_encoder_blocks"]):
        multi_head_block = MultiHeadAttentionBlock(config["d_model"], config["heads"], config["dropout"])
        feed_forward_block = FeedForwardBlock(config["d_model"], config["d_ff"], config["dropout"])
        encoder_blocks.append(EncoderBlock(multi_head_block,feed_forward_block))
    decoder_blocks = []
    for _ in range(config["num_decoder_blocks"]):
        self_attention_block = MultiHeadAttentionBlock(config["d_model"], config["heads"], config["dropout"])
        cross_attention_block = MultiHeadAttentionBlock(config["d_model"], config["heads"], config["dropout"])
        feed_forward_block = FeedForwardBlock(config["d_model"], config["d_ff"], config["dropout"])
        decoder_blocks.append(DecoderBlock(self_attention_block, cross_attention_block, feed_forward_block))
    
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    project = ProjectionLayer(config["d_model"], tokenizer_tgt.get_vocab_size())
    transformer = Transformer(src_embed, tgt_embed, src_pos, tgt_pos, encoder, decoder, project)
    return transformer
