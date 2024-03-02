from dataloader import get_dataloaders
from tokenizer import get_tokenizers
import lightning as L
from configuration import get_config
from transformer_model import build_transformer


if __name__ == "__main__":
    config = get_config()
    tokenizer_src, tokenizer_tgt = get_tokenizers()
    train_loader, val_loader = get_dataloaders(tokenizer_src, tokenizer_tgt)



    model = build_transformer(tokenizer_src, tokenizer_tgt)

    trainer = L.Trainer(max_epochs=config["num_epochs"], accelerator=config["accelerator"], devices=config["devices"])
    trainer.fit(model, train_loader, val_loader)