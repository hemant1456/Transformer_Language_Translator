from dataloader import get_dataloaders
from tokenizer import get_tokenizers
import lightning as L
from configuration import get_config
from transformer_model import build_transformer
from lightning.pytorch.callbacks import ModelCheckpoint
import os


if __name__ == "__main__":
    
    config = get_config()
    os.makedirs(config["model_weights_directory"],exist_ok=True)
    with open("train_log.txt",'w') as f:
        f.write("Starting Model Training")
        for key, value in config.items():
            print(f"{key:15}{str(value):15}")
            f.write(f"{key:15}{str(value):15}"+"\n")
    
    tokenizer_src, tokenizer_tgt = get_tokenizers()
    train_loader, val_loader = get_dataloaders(tokenizer_src, tokenizer_tgt)



    model = build_transformer(tokenizer_src, tokenizer_tgt)

    trainer = L.Trainer(max_epochs=config["num_epochs"], accelerator=config["accelerator"], devices=config["devices"],
                        callbacks=[ModelCheckpoint(config["model_weights_directory"])])
    trainer.fit(model, train_loader, val_loader)
