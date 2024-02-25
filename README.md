# Transformer Language Translator
This is a Transformer model designed to translate between two languages, it contains both encoder and decoder.

## Repository Files Information

### Core Files

| File         | Description                                                |
|--------------|------------------------------------------------------------|
| `main.py`    | You can directly use this file to train the model.         |
| `config.py`  | It contains the model architecture, make changes here for batch size, etc. |

### Data Handling

| File                | Description                                           |
|---------------------|-------------------------------------------------------|
| `tokenizer.py`      | Handles tokenization tasks.                           |
| `dataset_class.py`  | Handles dataset preparation tasks.                    |
| `dataloader.py`     | Handles data loading tasks.                           |

### Model Components

| File                      | Description                                                    |
|---------------------------|----------------------------------------------------------------|
| `embedding.py`            | Contains code for input and positional embeddings.             |
| `model_components.py`     | Contains building blocks of our transformer model such as FeedForwardBlock, ResidualConnection. |
| `multi_head_attention.py` | Contains code for our attention model.                         |
| `encoder.py`              | Contains code for encoder blocks.                              |
| `decoder.py`              | Contains code for decoder blocks.                              |
| `transformer.py`          | Contains code for our transformer class.                       |

### Miscellaneous

| File               | Description                                 |
|--------------------|---------------------------------------------|
| `requirements.txt` | Contains our dependencies.                  |

