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

def correct_spaces(text_ids, tokenizer, model, device, batch_size=32):
    #correct spaces
    new_text_ids = text_ids[0]
    
    dataX = torch.cat([text_ids] * (text_ids.size(1) - 2))

    for j in range(0, text_ids.size(1) - 2):
        dataX[j, j + 1] = tokenizer.mask_token_id

    with torch.no_grad():
        batch_probs = []
        for i in range(0, text_ids.size(1) - 2, batch_size):
            token_logits = model(dataX[i:i + batch_size, :].to(device))[0]
            token_logits = torch.softmax(token_logits, dim=2)

            mask_token_indexes = torch.where(dataX[i:i + batch_size, :] == tokenizer.mask_token_id)
            probs = token_logits[mask_token_indexes[0], mask_token_indexes[1], text_ids[:, mask_token_indexes[1]]]

            all_probs=token_logits[mask_token_indexes[0], mask_token_indexes[1], :]


            
            xs=[]
            for x in tokenizer.convert_ids_to_tokens(text_ids[:, mask_token_indexes[1]][0]):
                
                if x[0]=='▁':
                    new_x=x[1:]
                else:
                    new_x='▁'+x
                
                # print(x, new_x, new_x in tokenizer.get_vocab())
                
                if new_x not in tokenizer.get_vocab():
                    new_x=x
                xs.append(new_x)

            
            x2=tokenizer.convert_tokens_to_ids(xs)
            # print(xs)
            # print(x2)

            probs2 = token_logits[mask_token_indexes[0], mask_token_indexes[1], torch.tensor([x2])]


            c = probs2[0] > probs[0]

            x2=torch.tensor(x2)


            mask_token_indexes2=mask_token_indexes[1]
            # print(new_text_ids)
            new_text_ids[mask_token_indexes2[c]]=x2[c]
            # print(new_text_ids)
            
            # probs = probs[0].cpu().numpy()
            # batch_probs.append(probs)

            
        new_text=tokenizer.decode(new_text_ids[1:-1])

    return new_text
