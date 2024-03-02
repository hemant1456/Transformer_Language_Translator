from dataloader import get_dataloaders
from tokenizer import get_tokenizers
import lightning as L
from configuration import get_config
from transformer_model import build_transformer
from lightning.pytorch.callbacks import ModelCheckpoint
import os
import torch


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"]="true"
    
    config = get_config()
    os.makedirs(config["model_weights_directory"],exist_ok=True)
    with open("training_logs.txt",'w') as f:
        f.write("-----Starting Model Training----- \n")
        for key, value in config.items():
            print(f"{key:25}{str(value):15}")
            f.write(f"{key:25}{str(value):15}"+"\n")
        f.write("\n\n\n")
    
    tokenizer_src, tokenizer_tgt = get_tokenizers()
    train_loader, val_loader = get_dataloaders(tokenizer_src, tokenizer_tgt)
    torch.set_float32_matmul_precision("medium")


    model = build_transformer(tokenizer_src, tokenizer_tgt)

    trainer = L.Trainer(max_epochs=config["num_epochs"], accelerator=config["accelerator"], devices=config["devices"],
                        callbacks=[ModelCheckpoint(config["model_weights_directory"])],limit_val_batches=10)
    trainer.fit(model, train_loader, val_loader)
