from typing import Dict, Iterator
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from pathlib import Path
def load_or_build_tokenizer(tokenizer_file: str, ds_i: Iterator, lang: str) ->Tokenizer:
    ''' 
    Builds a Tokenizer if it doesn't exist else it loads already build tokenizer
    :param tokenizer_file : the file from which tokenizer will be loaded or saved if tokenizer does not exists
    :param ds_i: sentence iterator of dataset
    :lang : language code
    :returns: tokenizer
    '''
    tokenizer_path = Path(".")/ tokenizer_file.format(lang)
    if tokenizer_path.exists():
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(min_frequency=2, show_progress= True, special_tokens = ["[UNK]","[PAD]","[SOS]","[EOS]"])
        tokenizer.train_from_iterator(ds_i, trainer)
        tokenizer.save(str(tokenizer_path))
    return tokenizer

def get_sentence_iterator(ds, lang):
    '''
    Generator to yield sentences of a particular language one by one
    :param ds: dataset
    :param lang: language code 
    :return: yields sentence one by one
    '''
    for item in ds:
        yield item['translation'][lang]