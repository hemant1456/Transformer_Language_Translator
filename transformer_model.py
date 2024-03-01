from lightning import LightningModule

class Transformer(LightningModule):
    def __init__(self, src_embed, tgt_embed, src_pos, tgt_pos, encoder, decoder, project):
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