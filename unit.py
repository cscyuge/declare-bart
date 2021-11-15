import pickle

import numpy as np
import torch
import torch.nn.functional as F
from importlib import import_module
from transformers import BartTokenizer


def unlikelihood_loss(decoder_input_ids, logits, weight_mask, selective_penalty=True):
    """
    decoder_input_ids - (N, s)
    logits      - (N, s, vocab_size)
    weight_mask - (vocab_size,)
    """
    probs = F.softmax(logits, dim=-1)
    neg_probs = 1 - probs

    # replace zeros with small positive constant for stability
    neg_probs += (neg_probs == 0).float() * 1e-8
    log_neg_probs = torch.log(neg_probs)  # (N,s,v)

    # now create attention mask and apply it
    attention_mask = decoder_input_ids.eq(1).eq(0).float()
    attention_mask = attention_mask.unsqueeze(2).expand(-1, -1, logits.shape[2])
    log_neg_probs_masked = log_neg_probs * attention_mask

    # apply weight vector to the log probability tensor
    N, s = logits.size()[:2]
    weight_mask_expanded = weight_mask.unsqueeze(0).unsqueeze(0).expand(N, s, -1)
    weighted_probs = log_neg_probs_masked * weight_mask_expanded

    if selective_penalty:
        indices = torch.argmax(logits, dim=-1)
        indices_mask = F.one_hot(indices, num_classes=logits.shape[-1])  # (N,s,v)
        weighted_probs *= indices_mask

    # TODO: take into account batch size (doesn't matter now since N=1)
    return -torch.sum(weighted_probs)

if __name__ == '__main__':
    pretrained_model = 'facebook/bart-base'
    tokenizer = BartTokenizer.from_pretrained(pretrained_model)
    print(tokenizer.vocab_size)
    with open('./data/common_special.txt', 'r') as f:
        common = f.read().split('\n')
    weight_vector = np.zeros(tokenizer.vocab_size)
    ids = tokenizer.convert_tokens_to_ids(common)
    weight_vector[ids] = 1
    with open('./data/weight_vector.pkl','wb') as f:
        pickle.dump(weight_vector,f)

    # ul_loss = unlikelihood_loss(decoder_input_ids, lm_logits, weight_vector, unlikelihood_selective_penalty)
    # ul_loss_weighted = ul_loss * 100
    # loss += ul_loss_weighted