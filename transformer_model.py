from lightning import LightningModule
from embedding import InputEmbedding, PositionalEmbedding
from configuration import get_config
from encoder import Encoder, EncoderBlock
from multiheadattention import MultiHeadAttentionBlock
from model_component import FeedForwardBlock, ProjectionLayer
from decoder import Decoder, DecoderBlock
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import greedy_decode
import torch
import random

config = get_config()

class Transformer(LightningModule):
    def __init__(self, src_embed, tgt_embed, src_pos, tgt_pos, encoder, decoder, project,tokenizer_tgt):
        super().__init__()
        self.src_embed= src_embed
        self.tgt_embed= tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.encoder = encoder
        self.decoder= decoder
        self.projection= project
        self.pad_token = tokenizer_tgt.token_to_id("[PAD]")
        self.tokenizer_tgt = tokenizer_tgt
        self.tgt_vocab_size = tokenizer_tgt.get_vocab_size()
    def training_step(self, batch, batch_idx):
        enc_inputs, dec_inputs, src_mask, tgt_mask, labels, src_texts, tgt_texts = batch 
        encoder_output = self.encode(enc_inputs, src_mask)
        x = self.decode(dec_inputs, encoder_output, src_mask, tgt_mask)
        x = self.project(x)
        loss = F.cross_entropy(x.view(-1, self.tgt_vocab_size), labels.view(-1), ignore_index= self.pad_token)
        self.log("train_loss",loss, prog_bar=True, on_step=True)
        return loss
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr= config['lr'])
        return optimizer
    def encode(self,x, src_mask):
        x = self.src_embed(x)
        x = self.tgt_pos(x)
        x = self.encoder(x, src_mask)
        return x
    def decode(self, dec_inputs, encoder_output, src_mask, tgt_mask):
        x= self.tgt_embed(dec_inputs)
        x = self.tgt_pos(x)
        x = self.decoder(x, encoder_output, src_mask, tgt_mask)
        return x
    def project(self,x):
        x = self.projection(x)
        return x
    def on_validation_epoch_start(self):
        self.src_texts = []
        self.tgt_texts = []
        self.predicted_texts =[]
    def validation_step(self, batch, batch_idx):
        enc_inputs, dec_inputs, src_mask, tgt_mask, labels, src_texts, tgt_texts = batch
        self.src_texts.append(src_texts[0])
        self.tgt_texts.append(tgt_texts[0])
        predicted_text = greedy_decode(self, self.tokenizer_tgt, enc_inputs, src_mask)
        self.predicted_texts.append(predicted_text)
    def on_validation_epoch_end(self):
        for _ in range(2):
            idx = random.randint(0,len(self.src_texts)-1)
            with open("training_logs.txt","a") as f:
                f.write(f"SOURCE: {self.src_texts[idx]}"+"\n")
                f.write(f"TARGET: {self.tgt_texts[idx]}"+"\n")
                f.write(f"PREDICTED {self.predicted_texts[idx]}"+"\n\n\n")
    def on_train_epoch_start(self):
        with open("training_logs.txt","a") as f:
                f.write(f"The current learning rate is : {self.trainer.optimizers[0].param_groups[0]['lr']}"+"\n")
    def on_train_epoch_end(self):
        with open("training_logs.txt","a") as f:
                f.write(f"Current Epoch is : {self.current_epoch} and the loss is {self.trainer.callback_metrics['train_loss']}"+"\n")
    
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
    transformer = Transformer(src_embed, tgt_embed, src_pos, tgt_pos, encoder, decoder, project,tokenizer_tgt)
    return transformer
