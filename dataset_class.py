import torch
import lightning as L

def causal_mask(size):
    '''
    this is required for attention mask so that our decode does not see future tokens
    '''
    return torch.triu(torch.ones((size,size)),diagonal=1)==0

class BilingualDataset(L.LightningDataModule):
    '''
    for encoder input will look like this [start_of_sentence_token, sentence_tokens, end_of_sentence_token, padding_tokens]
    for decoder input will look like this [start_of_sentence_token, sentence_tokens, padding_tokens]
    for label label will look like this [sentence_tokens, end_of_sentence_token, padding_tokens]
    '''
    def __init__(self, config, dataset, tokenizer_src, tokenizer_tgt):
        self.ds = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.config = config
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)
        self.seq_len = config["seq_len"]
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, index):
        src_tgt_pair = self.ds[index]
        src_text = src_tgt_pair["translation"][self.config["lang_src"]]
        tgt_text = src_tgt_pair["translation"][self.config["lang_tgt"]]

        src_tokens = self.tokenizer_src.encode(src_text).ids
        tgt_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_padding_tokens = self.seq_len - 2 - len(src_tokens)
        dec_padding_tokens = self.seq_len - 1 - len(tgt_tokens)

        assert enc_padding_tokens>0 and dec_padding_tokens>0, "sentence length is too long"

        enc_input = torch.cat([
            self.sos_token,
            torch.tensor(src_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_padding_tokens)
            
        ]).unsqueeze(0)
        dec_input = torch.cat([
            self.sos_token,
            torch.tensor(tgt_tokens),
            torch.tensor([self.pad_token] * dec_padding_tokens)
        ]).unsqueeze(0)

        label = torch.cat([
            torch.tensor(tgt_tokens),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_padding_tokens)
        ]).unsqueeze(0)
        
        enc_mask = enc_input!=self.pad_token

        dec_mask = dec_input!=self.pad_token & causal_mask(enc_input.size(1))

        assert len(enc_input)==len(dec_input)==len(label)

        return enc_input, dec_input, label, enc_mask, dec_mask


