from pathlib import Path
import torch

def get_config():
    '''
    this will contain hyperparameters for our transformer model
    it also defines source and target language for our dataset
    '''
    return {
        "min_frequency":1, # minimum frequency for tokenzier to include in vocab
        "batch_size": 32, 
        "num_epochs": 20,
        "base_lr": 1e-4, #base learning rate our one cycle epoch, max_lr will be 10* base_lr in our case
        "heads": 8, #number of heads
        "seq_len": 165,
        "d_model": 512,
        "d_ff": 512, #dimension_feed_forward
        "dropout": 0.1,
        "num_encoder_blocks":8,
        "num_decoder_blocks":8,
        "accelerator": "cuda" if torch.cuda.is_available() else "cpu",
        "devices": 1,
        "lang_src": 'en',
        "lang_tgt": 'fr',
        "model_weights_directory": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{}.json",
        "experiment_name": "runs/tmodel",
        "train_num_workers":5
    }

