import torch
import lightning as L

def causal_mask(size):
    '''
    this is required for attention mask so that our decode does not see future tokens
    '''
    return torch.tril(torch.ones((size,size)))

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
        self.seq_len = config["seq_len"]
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, index):
        src_tgt_pair = self.ds[index]
        src_text = src_tgt_pair["translation"][self.config["lang_src"]]
        tgt_text = src_tgt_pair["translation"][self.config["lang_tgt"]]
        

        return src_text, tgt_text
