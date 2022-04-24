import torch
from torch import nn


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def init_weights(model, model_state_filepath=None):
    if not model_state_filepath:
        if model.name == 'transformer':
            init_func = nn.init.xavier_uniform_
        else:
            init_func = nn.init.uniform_

        for name, param in model.named_parameters():
            init_func(param, -0.08, 0.08)
    else:
        state_dict = torch.load(model_state_filepath)
        print(f'Loading model state from {model_state_filepath}')
        model.load_state_dict({
            state: value
            for state, value in state_dict.items()
            if state in model.state_dict()
        })


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_mask(src, tgt, device, pad_idx):
    """
    We create a ``subsequent word`` mask to stop a target word from
    attending to its subsequent words. We also create masks, for masking
    source and target padding tokens
    """
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def flatten(some_list):
    return [item for sub_list in some_list for item in sub_list]


def remove_tech_tokens(some_str, tokens_to_remove=('<eos>', '<sos>', '<unk>', '<pad>')):
    return [x for x in some_str if x not in tokens_to_remove]


def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
