import torch
def greedy_decode(model, tokenizer_tgt, enc_inputs, src_mask):
    i=0
    sos_id = tokenizer_tgt.token_to_id("[SOS]")
    eos_id = tokenizer_tgt.token_to_id("[EOS]")
    prediction = ""
    proj_ids = [sos_id]
    decoder_input = torch.tensor(proj_ids, dtype=torch.int64).unsqueeze(0)
    encoder_output = model.encode(enc_inputs, src_mask)
    
    while i<160:
        x = model.decode(decoder_input, encoder_output, src_mask, None)
        x = model.project(x)
        x = x.view(-1).argmax()
        x= tokenizer_tgt.id_to_token(x.item())
        if x==eos_id:
            break
        proj_ids.append(x)
        prediction+= " " + x
        i+=1
    return prediction