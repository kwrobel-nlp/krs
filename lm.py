import numpy as np
import torch


def text_prob(text_ids, mask_token_id, model, device):
    dataX = torch.cat([text_ids] * (text_ids.size(1) - 2))
    for j in range(0, text_ids.size(1) - 2):
        dataX[j, j + 1] = mask_token_id

    with torch.no_grad():
        token_logits = model(dataX.to(device))[0]
        token_logits = torch.softmax(token_logits, dim=2)
        mask_token_indexes = torch.where(dataX == mask_token_id)
        probs = token_logits[mask_token_indexes[0], mask_token_indexes[1], text_ids[:, mask_token_indexes[1]]]
        probs = probs[0].cpu().numpy()
        logprob = np.sum(np.log(probs))
        logprob_wo0 = np.sum(np.log(probs[probs > 0]))

    return logprob.item(), text_ids.size(1) - 2, logprob_wo0.item()
