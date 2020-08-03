import numpy as np
import torch


def text_prob(text_ids, mask_token_id, model, device, batch_size=32):
    dataX = torch.cat([text_ids] * (text_ids.size(1) - 2))

    for j in range(0, text_ids.size(1) - 2):
        dataX[j, j + 1] = mask_token_id

    with torch.no_grad():
        batch_probs = []
        for i in range(0, text_ids.size(1) - 2, batch_size):
            token_logits = model(dataX[i:i + batch_size, :].to(device))[0]
            token_logits = torch.softmax(token_logits, dim=2)

            mask_token_indexes = torch.where(dataX[i:i + batch_size, :] == mask_token_id)
            probs = token_logits[mask_token_indexes[0], mask_token_indexes[1], text_ids[:, mask_token_indexes[1]]]
            probs = probs[0].cpu().numpy()
            batch_probs.append(probs)

        probs = np.concatenate(batch_probs)
        logprob = np.sum(np.log(probs))
        logprob_wo0 = np.sum(np.log(probs[probs > 0]))

    return logprob.item(), text_ids.size(1) - 2, logprob_wo0.item()
