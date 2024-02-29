from dataset_class import BilingualDataset, causal_mask
import torch
from tokenizer import get_tokenizers
from config import get_config
from datasets import load_dataset

config = get_config()
tokenizer_src, tokenizer_tgt = get_tokenizers()

sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)



def dynamic_padding_collate_fn(batch):
    '''
    this function dynamically pads based on maximum input size thus saving lots of computation
    returns: encoder_mask of shape (batch_size, 1, 1, seq_len), decoder_mask of shape (batch_size, 1, seq_len, seq_len)
    '''
    max_length = 0 
    for src_text, tgt_text in batch:
        src_tokens = tokenizer_src.encode(src_text).ids
        tgt_tokens = tokenizer_tgt.encode(tgt_text).ids
        max_length = max(max_length, len(src_tokens), len(tgt_tokens))
    
    
    enc_inputs = []
    dec_inputs = []
    labels = []
    src_texts = []
    tgt_texts = []
    
    for src_text, tgt_text in batch:
        src_texts.append(src_text)
        tgt_texts.append(tgt_text)

        src_tokens = tokenizer_src.encode(src_text).ids
        tgt_tokens = tokenizer_tgt.encode(tgt_text).ids
        enc_padding_tokens = max_length - len(src_tokens) 
        dec_padding_tokens = max_length +1 - len(tgt_tokens) 

        enc_input = torch.cat([
                sos_token,
                torch.tensor(src_tokens, dtype=torch.int64),
                eos_token,
                torch.tensor([pad_token] * enc_padding_tokens, dtype=torch.int64)
                
            ])
        enc_inputs.append(enc_input)

        dec_input = torch.cat([
                sos_token,
                torch.tensor(tgt_tokens),
                torch.tensor([pad_token] * dec_padding_tokens, dtype=torch.int64)
            ])
        dec_inputs.append(dec_input)
        
        label = torch.cat ([
                torch.tensor(tgt_tokens, dtype= torch.int64),
                eos_token,
                torch.tensor([pad_token]* dec_padding_tokens, dtype = torch.int64)
            ])
        labels.append(label)

    enc_inputs = torch.stack(enc_inputs,dim=0)
    dec_inputs = torch.stack(dec_inputs, dim=0)
    encoder_mask = (enc_inputs != pad_token).unsqueeze(1).unsqueeze(1).int()
    #1 1 52 52
    #8 1  1 52
    decoder_mask = (causal_mask(dec_inputs.size(1)) & (dec_inputs != pad_token).unsqueeze(1)).unsqueeze(1)

    labels = torch.stack(labels, dim=0)

    return enc_inputs, dec_inputs, encoder_mask, decoder_mask, labels, src_texts, tgt_texts

def get_max_seq_length(ds_raw):
    '''
    prints the maximum length of a sentence, it is crucial as it will decide the seq length of our model
    '''
    max_len_src = 0
    max_len_tgt = 0
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    
    print(f'Max length of source sentence : {max_len_src}')
    print(f'Max length of target senentece: {max_len_tgt}')

def data_cleanup(ds_raw):
    print(type(ds_raw))
    ds_raw = list(ds_raw)
    filtered_data = filter(lambda item: len(item['translation']['en'].split(" "))<=150 and len(item['translation']['fr'].split(" "))<=150,ds_raw)
    sorted_data = sorted(filtered_data, key = lambda item: len(item['translation']['en'].split(" ")))
    return sorted_data

def get_dataloaders():
    '''
    simple functions to get dataloaders
    we have divided our dataset into 90% train and 10% validation
    for validation we will be taking 1 batch at a time
    '''
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')
    
    get_max_seq_length(ds_raw)
    train_ds_size = int(0.9* len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size

    train_ds_raw, val_ds_raw = torch.utils.data.random_split(ds_raw, [train_ds_size, val_ds_size])
    train_ds_raw = data_cleanup(train_ds_raw)
    val_ds_raw = data_cleanup(val_ds_raw)
    train_ds = BilingualDataset(config, train_ds_raw, tokenizer_src, tokenizer_tgt)
    val_ds = BilingualDataset(config, val_ds_raw, tokenizer_src, tokenizer_tgt)

    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=config['batch_size'], num_workers=5,persistent_workers=True,pin_memory=True, shuffle=False, collate_fn=dynamic_padding_collate_fn)
    val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=True, collate_fn=dynamic_padding_collate_fn)

    return train_dataloader, val_dataloader

