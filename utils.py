import torch
from configuration import get_config
from dataset_class import causal_mask
config = get_config()
def greedy_decode(model, tokenizer_tgt, source, source_mask, max_len=160):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    device = source.device
    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = (causal_mask(decoder_input.size(1))==0).unsqueeze(0).unsqueeze(0).to(device)

        out = model.decode(decoder_input, encoder_output, source_mask, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return tokenizer_tgt.decode(decoder_input.squeeze(0)[1:-1].detach().cpu().numpy())

import torch
import numpy as np

def beam_search(model, tokenizer_tgt, source, source_mask, max_len=160, beam_size=5):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    device = source.device

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)

    # Initialize the beam with the start-of-sentence token
    beams = [(torch.tensor([sos_idx], device=device), 0.0)]  # Each beam is a tuple (sequence, score)

    for _ in range(max_len):
        candidates = []
        for seq, score in beams:
            # Check if the sequence ends with EOS
            if seq[-1] == eos_idx:
                alpha = 0.6
                length_penalty = ((5 + seq.size(0)) ** alpha) / (6 ** alpha)  # Length penalty
                candidates.append((seq, score / length_penalty))
                continue

            # Prepare the input for the decoder
            decoder_input = seq.unsqueeze(0)  # Add batch dimension
            decoder_mask = (causal_mask(decoder_input.size(1))==0).unsqueeze(0).unsqueeze(0).to(device)

            # Decode the sequence
            out = model.decode(decoder_input, encoder_output, source_mask, decoder_mask)
            prob = model.project(out[:, -1])  # Project the output to vocabulary space

            # Convert logit to probabilities and take the top beam_size candidates
            log_prob = torch.log_softmax(prob, dim=-1)
            top_log_probs, top_idxs = torch.topk(log_prob, beam_size)

            # Add new candidates to consider
            for i in range(beam_size):
                next_seq = torch.cat([seq, top_idxs[0][i].unsqueeze(0)])
                next_score = score + top_log_probs[0][i].item()  # Accumulate log probability
                candidates.append((next_seq, next_score))

        # Keep top beam_size sequences
        ordered = sorted(candidates, key=lambda x: x[1], reverse=True)
        beams = ordered[:beam_size]

    # Choose the best sequence
    best_seq, best_score = beams[0]
    return tokenizer_tgt.decode(best_seq.cpu().numpy())
