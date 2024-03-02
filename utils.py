import torch
from configuration import get_config
config = get_config()

def greedy_decode(model, tokenizer_tgt, enc_inputs, src_mask, max_lengths=160):
    i=0
    sos_id = tokenizer_tgt.token_to_id("[SOS]")
    eos_id = tokenizer_tgt.token_to_id("[EOS]")
    prediction = []
    proj_ids = [sos_id]
    
    encoder_output = model.encode(enc_inputs, src_mask)
    
    for _ in range(160):
        decoder_input = torch.tensor(proj_ids, dtype=torch.int64).unsqueeze(0).to(torch.device(config["accelerator"]))
        decoder_output = model.decode(decoder_input, encoder_output, src_mask, None)
        decoder_projection= model.project(decoder_output)
        last_token_id = (decoder_projection[:,-1,:].view(-1).argmax()).item()
        last_token= tokenizer_tgt.id_to_token(last_token_id)
        if last_token_id==eos_id:
            break
        proj_ids.append(last_token_id)
        prediction.append(last_token)
        i+=1
    return " ".join(prediction)
