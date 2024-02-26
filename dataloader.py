from dataset_class import BilingualDataset
import torch
from tokenizer import load_or_build_tokenizer, get_sentence_iterator
from datasets import load_dataset 
from config import get_config

config = get_config()
ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')

dsi_src = get_sentence_iterator(ds_raw,config["lang_src"])
tokenizer_src = load_or_build_tokenizer(config['tokenizer_file'], dsi_src, config['lang_src'])

dsi_tgt = get_sentence_iterator(ds_raw,config["lang_tgt"])
tokenizer_tgt = load_or_build_tokenizer(config['tokenizer_file'], dsi_tgt, config["lang_tgt"])

def get_max_seq_length(ds_raw):
    '''
    prints the maximum length of a sentence, it is crucial as it will decide the seq length of our model
    '''
    max_len_src = 0
    max_len_tgt = 0
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_src.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    
    print(f'Max length of source sentence : {max_len_src}')
    print(f'Max length of target senentece: {max_len_tgt}')


def get_dataloaders(config, ds_raw, tokenizer_src, tokenizer_tgt):
    '''
    simple functions to get dataloaders
    we have divided our dataset into 90% train and 10% validation
    for validation we will be taking 1 batch at a time
    '''
    get_max_seq_length(ds_raw)

    train_ds_size = int(0.9* len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size

    train_ds_raw, val_ds_raw = torch.utils.data.random_split(ds_raw, [train_ds_size, val_ds_size])
    train_ds = BilingualDataset(config, train_ds_raw, tokenizer_src, tokenizer_tgt)
    val_ds = BilingualDataset(config, val_ds_raw, tokenizer_src, tokenizer_tgt)

    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader
    