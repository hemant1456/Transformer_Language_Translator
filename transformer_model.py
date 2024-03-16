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
from utils import greedy_decode, beam_search
import torch
import random
import torch.nn.init as init
import sacrebleu


config = get_config()

def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)


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
        loss = F.cross_entropy(x.view(-1, self.tgt_vocab_size), labels.view(-1), ignore_index= self.pad_token, label_smoothing=0.1)
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr= config['base_lr'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr= 10 * config["base_lr"], total_steps=None, epochs=config["num_epochs"], steps_per_epoch=3565, pct_start=0.2, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=10, final_div_factor=10, three_phase=False, last_epoch=-1, verbose='deprecated')
        return {"optimizer":optimizer, "lr_scheduler": {"scheduler":scheduler, "interval":"step"}}
    def encode(self,x, src_mask):
        x = self.src_embed(x)
        x = self.src_pos(x)
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
        self.greedy_predicted_texts =[]
        self.beam_predicted_texts =[]
    def validation_step(self, batch, batch_idx):
        enc_inputs, dec_inputs, src_mask, tgt_mask, labels, src_texts, tgt_texts = batch
        self.src_texts.append(src_texts[0])
        self.tgt_texts.append(tgt_texts[0])
        greedy_predicted_text = greedy_decode(self, self.tokenizer_tgt, enc_inputs, src_mask[:,:,0,:])
        beam_predicted_text = beam_search(self, self.tokenizer_tgt, enc_inputs, src_mask[:,:,0,:])
        self.greedy_predicted_texts.append(greedy_predicted_text)
        self.beam_predicted_texts.append(beam_predicted_text)
    def on_validation_epoch_end(self):
        tgt_texts_flat = [" ".join(text) for text in self.tgt_texts]
        predicted_texts_flat = [" ".join(text) for text in self.beam_predicted_texts]

        # Calculate BLEU score
        bleu = sacrebleu.corpus_bleu(predicted_texts_flat, [tgt_texts_flat])
        self.log('val_bleu', bleu.score, on_epoch=True, prog_bar=True)
        idx = random.sample(range(0,len(self.src_texts)), 2)
        for i in range(2):
            
            with open("training_logs.txt","a") as f:
                f.write(f"[SOURCE]: {self.src_texts[idx[i]]}"+"\n")
                f.write(f"[TARGET]: {self.tgt_texts[idx[i]]}"+"\n")
                f.write(f"[greedy_PREDICTED] {self.greedy_predicted_texts[idx[i]]}"+"\n")
                f.write(f"[beam_PREDICTED] {self.beam_predicted_texts[idx[i]]}"+"\n")
                f.write(f"[Beam] bleu score is {bleu} \n\n\n")
    def on_train_epoch_start(self):
        with open("training_logs.txt","a") as f:
                f.write(f"The current learning rate is : {self.trainer.optimizers[0].param_groups[0]['lr']:.6f}"+"\n")
    def on_train_epoch_end(self):
        with open("training_logs.txt","a") as f:
                f.write(f"Current Epoch is : {self.current_epoch} and the loss is {self.trainer.callback_metrics['train_loss']:.2f}"+"\n")
    
def build_transformer(tokenizer_src, tokenizer_tgt):
    src_embed = InputEmbedding(config["d_model"], tokenizer_src.get_vocab_size())
    tgt_embed = InputEmbedding(config["d_model"], tokenizer_tgt.get_vocab_size())
    src_pos = PositionalEmbedding(config["d_model"], config["seq_len"])
    tgt_pos = PositionalEmbedding(config["d_model"], config["seq_len"])

    encoder_blocks = []
    for _ in range(config["num_encoder_blocks"]//2):
        multi_head_block = MultiHeadAttentionBlock(config["d_model"], config["heads"], config["dropout"])
        feed_forward_block = FeedForwardBlock(config["d_model"], config["d_ff"], config["dropout"])
        encoder_blocks.append(EncoderBlock(multi_head_block,feed_forward_block))
    e1, e2, e3, e4 = encoder_blocks
    encoder_blocks_shared= [e1, e2, e3, e4, e4, e3, e2, e1]
    decoder_blocks = []
    for _ in range(config["num_decoder_blocks"]//2):
        self_attention_block = MultiHeadAttentionBlock(config["d_model"], config["heads"], config["dropout"])
        cross_attention_block = MultiHeadAttentionBlock(config["d_model"], config["heads"], config["dropout"])
        feed_forward_block = FeedForwardBlock(config["d_model"], config["d_ff"], config["dropout"])
        decoder_blocks.append(DecoderBlock(self_attention_block, cross_attention_block, feed_forward_block))
    
    d1, d2 , d3, d4 = decoder_blocks
    decoder_blocks_shared = [d1, d2, d3, d4, d4, d3, d2, d1]
    encoder = Encoder(nn.ModuleList(encoder_blocks_shared))
    decoder = Decoder(nn.ModuleList(decoder_blocks_shared))

    project = ProjectionLayer(config["d_model"], tokenizer_tgt.get_vocab_size())
    transformer = Transformer(src_embed, tgt_embed, src_pos, tgt_pos, encoder, decoder, project,tokenizer_tgt)
    transformer.apply(initialize_weights)
    

    return transformer
