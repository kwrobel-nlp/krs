import collections
import json
import time

import jsonlines as jsonlines

from reader import load_both
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from difflib import SequenceMatcher
import numpy as np
import tqdm

#faster

device='cuda'
device='cpu'

print(torch.cuda.is_available())

model_name="xlm-roberta-base"
data = load_both('data/omg/reference.txt', 'data/omg/nbest.txt')

def get_logits(text_ids1, masked1, model, tokenizer):
  with torch.no_grad():
    token_logits = model(masked1.to(device))[0]
  mask_token_indexes = torch.where(masked1 == tokenizer.mask_token_id)[1]
  #print(mask_token_indexes)
  token_logits=torch.softmax(token_logits, dim=2)
  #print(torch.sum(token_logits[0, 0, :]))
  return token_logits[0, mask_token_indexes, text_ids1[:, mask_token_indexes]]
  #return token_logits[0, mask_token_indexes, text_ids1[:, mask_token_indexes]]

  mask_token_logits=[]
  for pos, token_index in enumerate(mask_token_indexes):
    #print(token_index, text_ids1[:, token_index])
    token_logit=token_logits[0, token_index, text_ids1[:, token_index]]
    #print(token_logit)
    mask_token_logits.append(token_logit)
  #print(mask_token_logits)
  return mask_token_logits

def gmean(input_x, dim):
  log_x = torch.log(input_x)
  return torch.exp(torch.mean(log_x, dim=dim))


flatten = lambda l: [item for sublist in l for item in sublist]


tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.eval()
if device=='cuda':
  model.half()
  model.to(device)
  model.half()

start=time.time()
count=0
with jsonlines.open(model_name+'output1.jsonl', mode='w', flush=True) as writer:
  
  for id, utt in tqdm.tqdm(data.items(), desc="Texts"):
    texts = utt['candidates']
    #texts=texts[:10]
  
    texts_ids=[]
    for text in texts:
      ids = tokenizer.encode(text, return_tensors="pt")
      texts_ids.append(ids)
  
    utt['preds'] = collections.defaultdict(dict)
    uniq_masks = set()
    for i, text_ids1 in enumerate(tqdm.tqdm(texts_ids, desc="Cands")):
      for j, text_ids2 in enumerate(texts_ids):
        if i <= j: continue
        
        count+=1
        text_ids1l = text_ids1.tolist()[0]
        text_ids2l = text_ids2.tolist()[0]
        s = SequenceMatcher(None, text_ids1l, text_ids2l)
        # stworz tensory wypelnione maską i naloz pasujace bloki
        masked1 = torch.tensor(np.full_like(text_ids1, tokenizer.mask_token_id))
        # print(text_ids1.size())
        # print(masked1.size())
        masked2 = torch.tensor(np.full_like(text_ids2, tokenizer.mask_token_id))
  
        for block in s.get_matching_blocks():
          
          
          
          masked1[:, block.a:(block.a + block.size)] = torch.tensor(text_ids1l[block.a:(block.a + block.size)])
          masked2[:, block.b:(block.b + block.size)] = torch.tensor(text_ids2l[block.b:(block.b + block.size)])
  
        uniq_masks.add(tuple(masked1.tolist()[0]))
        uniq_masks.add(tuple(masked2.tolist()[0]))
        # continue
        # print(text_ids1.size())
        # print(masked1.size())
        # print(masked1, masked2)
  
        l1 = get_logits(text_ids1, masked1, model, tokenizer)
        l2 = get_logits(text_ids2, masked2, model, tokenizer)
  
        utt['preds'][i][j] = (flatten(l1.tolist()), flatten(l2.tolist()))
  
        # print(i, j, l1.tolist(), l2.tolist())
        # # gl1 = l1
        # # gl2 = l2
        # gl1 = gmean(l1, 1)
        # gl2 = gmean(l2, 1)
        # print(gl1, gl2)
        # 
        # if gl1 > gl2:
        #   print(f"{i} better than {j}")
        #   print(f"{texts[i]} THAN \n{texts[j]}")
        # else:
        #   print(f"{j} better than {i}")
        #   print(f"{texts[j]} THAN \n{texts[i]}")
  
    writer.write(utt)
    #break
      
  end = time.time()
  print(end-start, count)
  
  json.dump(data, open(model_name+'data_temp1.json','w'), ensure_ascii=False, indent=2)
