from pathlib import Path

def get_config():
    '''
    this will contain hyperparameters for our transformer model
    it also defines source and target language for our dataset
    '''
    return {
        "min_frequency":2, # minimum frequency for tokenzier to include in vocab
        "batch_size": 8, 
        "num_epochs": 20,
        "lr": 1e-4,
        "heads": 8, #number of heads
        "seq_len": 160,
        "d_model": 512,
        "d_ff": 248, #dimension_feed_forward
        "dropout": 0.1,
        "num_encoder_blocks":6,
        "num_decoder_blocks":6,
        "lang_src": 'en',
        "lang_tgt": 'fr',
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{}.json",
        "experiment_name": "runs/tmodel"
    }

