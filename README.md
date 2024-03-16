# Transformer Language Translator

## Introduction

This repository hosts the implementation of a Sequence to Sequence Transformer model, tailored for language translation between two languages. Building upon a foundational tutorial ([https://www.youtube.com/watch?v=ISNdQcPhsts&ab_channel=UmarJamil](#)), this project introduces significant optimizations and enhancements to improve efficiency, understandability, and performance.

### Key Enhancements

- **Dynamic Padding in DataLoader:** Introduced dynamic padding to significantly accelerate training by padding sequences based on the maximum length within a batch rather than a predefined maximum size, achieving a 5-6 times faster training speed.
- **PyTorch Lightning Integration:** Restructured the Transformer model to leverage PyTorch Lightning, simplifying the training process and facilitating the experimentation with various hyperparameters.
- **Parameter Sharing:** Implemented parameter sharing across different layers of the Transformer model, reducing the total number of parameters while enabling a deeper model architecture. This approach leads to better regularization and enhances the model's learning capability.
- **Performance Achievement:** Reached a BLEU score of approximately 41 on an English to French dataset, aligning with benchmarks set in the seminal "Attention Is All You Need" paper.

## Repository Structure

### Core Training Files

- `main.py`: The primary script for initiating model training.
- `configuration.py`: Defines the model's architecture and hyperparameters. Modify here for batch sizes and other parameters.

### Data Preparation and Handling

- `tokenizer.py`: Manages the tokenization process for preparing textual data.
- `dataset_class.py`: Facilitates dataset preparation and preprocessing tasks.
- `dataloader.py`: Manages the efficient loading of data, incorporating dynamic padding for optimized batch processing.

### Transformer Model Components

- `embedding.py`: Implements input and positional embeddings for the Transformer model.
- `model_components.py`: Contains the foundational components of the Transformer model, such as the FeedForwardBlock and ResidualConnection.
- `multi_head_attention.py`: Implements the multi-head attention mechanism of the Transformer model.
- `encoder.py`: Houses the encoder part of the Transformer architecture.
- `decoder.py`: Contains the decoder components of the Transformer model.
- `transformer.py`: Defines the complete Transformer model, integrating both the encoder and decoder components.

### Additional Resources

- `requirements.txt`: Lists the necessary dependencies for replicating the model's environment.

## Getting Started

To begin working with the Transformer Language Translator, clone this repository and ensure that all dependencies listed in `requirements.txt` are installed. For detailed instructions on training the model, refer to the usage guidelines provided in `main.py`.

## Contributions

Contributions to this project are welcome. If you're interested in improving the Transformer Language Translator or have suggestions for further enhancements, please feel free to submit a pull request or open an issue.

### Acknowledgments

This project builds on the instructional material provided in a specific tutorial video (referenced at the beginning of this document). I extend my gratitude to the creator of the original content Umar Jamil for helping me understand transformer model architecture. 