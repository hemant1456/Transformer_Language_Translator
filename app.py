# the gradio file
import gradio as gr
from tokenizer import get_tokenizers
from transformer_model import build_transformer
from utils import greedy_decode, beam_search
import torch

if __name__=="__main__":
    tokenizer_src, tokenizer_tgt = get_tokenizers()
    model = build_transformer(tokenizer_src, tokenizer_tgt)
    weights = torch.load("en-fr_translator_model_trained.pth", map_location="cpu")
    model.load_state_dict(weights.state_dict())


    pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64).to(torch.device("cpu"))
    sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64).to(torch.device("cpu"))
    eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64).to(torch.device("cpu"))

    model.eval()
    @torch.no_grad
    def translator(text):
        start_token = tokenizer_src.token_to_id("[SOS]")
        end_token = tokenizer_src.token_to_id("[EOS]")
        encoder_input= [start_token] + tokenizer_src.encode(text).ids + [end_token]
        encoder_input = torch.tensor(encoder_input, dtype=torch.int64).unsqueeze(0)
        beam_prediction = beam_search(model, tokenizer_tgt, encoder_input, None)
        return beam_prediction
    app = gr.Interface(translator, inputs= ["text"], outputs=["text"], 
                    examples= ["The gentle breeze of the early morning carried the sweet fragrance of jasmine, signaling the arrival of spring and the promise of new beginnings.",
                                "In the heart of the bustling city, a quaint coffee shop offers a haven of tranquility, where patrons find solace in the aroma of freshly brewed coffee and the warmth of shared stories."],
                    title="English to French language Translator")
    app.launch()