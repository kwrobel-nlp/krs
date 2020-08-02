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

#faster, but no memory

print(torch.cuda.is_available())

model_name="xlm-roberta-base"
data = load_both('data/test/reference.txt', 'data/test/nbest.txt')

def get_logits(text_ids1, masked1, model, tokenizer):
  #with torch.no_grad():
  #  token_logits = model(masked1)[0]
  token_logits=inferenced[tuple(masked1.tolist()[0])]
      
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
  print(mask_token_logits)
  return mask_token_logits

def gmean(input_x, dim):
  log_x = torch.log(input_x)
  return torch.exp(torch.mean(log_x, dim=dim))


flatten = lambda l: [item for sublist in l for item in sublist]


tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.eval()
model.to('cuda')

start=time.time()
count=0
with jsonlines.open(model_name+'output.jsonl', mode='w') as writer:
  
  for id, utt in tqdm.tqdm(data.items()):
    texts = utt['candidates']
    texts=texts[:10]
  
    texts_ids=[]
    for text in texts:
      ids = tokenizer.encode(text, return_tensors="pt")
      texts_ids.append(ids)
  
    utt['preds'] = collections.defaultdict(dict)
    uniq_masks = {}
    for i, text_ids1 in enumerate(texts_ids):
      for j, text_ids2 in enumerate(texts_ids):
        if i <= j: continue
        
        count+=1
        text_ids1l = text_ids1.tolist()[0]
        text_ids2l = text_ids2.tolist()[0]
        s = SequenceMatcher(None, text_ids1l, text_ids2l)
        # stworz tensory wypelnione maską i naloz pasujace bloki
        masked1 = torch.tensor(np.full_like(text_ids1, tokenizer.mask_token_id))
  
        masked2 = torch.tensor(np.full_like(text_ids2, tokenizer.mask_token_id))
  
        for block in s.get_matching_blocks():
          masked1[:, block.a:(block.a + block.size)] = torch.tensor(text_ids1l[block.a:(block.a + block.size)])
          masked2[:, block.b:(block.b + block.size)] = torch.tensor(text_ids2l[block.b:(block.b + block.size)])
  
        uniq_masks[tuple(masked1.tolist()[0])]=masked1
        uniq_masks[tuple(masked2.tolist()[0])]=masked2
        continue
  
    inferenced={}
    
    for tuple_masked1, masked1 in uniq_masks.items():
      with torch.no_grad():
        #print(masked1.size())
        token_logits = model(masked1.to('cuda'))[0]
      inferenced[tuple_masked1]=token_logits
  
                
    for i, text_ids1 in enumerate(texts_ids):
      for j, text_ids2 in enumerate(texts_ids):
        if i <= j: continue
  
        #count += 1
        text_ids1l = text_ids1.tolist()[0]
        text_ids2l = text_ids2.tolist()[0]
        s = SequenceMatcher(None, text_ids1l, text_ids2l)
        # stworz tensory wypelnione maską i naloz pasujace bloki
        masked1 = torch.tensor(np.full_like(text_ids1, tokenizer.mask_token_id))
  
        masked2 = torch.tensor(np.full_like(text_ids2, tokenizer.mask_token_id))
  
        for block in s.get_matching_blocks():
          masked1[:, block.a:(block.a + block.size)] = torch.tensor(text_ids1l[block.a:(block.a + block.size)])
          masked2[:, block.b:(block.b + block.size)] = torch.tensor(text_ids2l[block.b:(block.b + block.size)])
  
        # uniq_masks.add(tuple(masked1.tolist()[0]))
        # uniq_masks.add(tuple(masked2.tolist()[0]))
  
        l1 = get_logits(text_ids1, masked1, model, tokenizer)
        l2 = get_logits(text_ids2, masked2, model, tokenizer)
  
        utt['preds'][i][j] = (flatten(l1.tolist()), flatten(l2.tolist()))
  
        print(i, j, l1.tolist(), l2.tolist())
        # gl1 = l1
        # gl2 = l2
        gl1 = gmean(l1, 1)
        gl2 = gmean(l2, 1)
        print(gl1, gl2)
  
        if gl1 > gl2:
          print(f"{i} better than {j}")
          print(f"{texts[i]} THAN \n{texts[j]}")
        else:
          print(f"{j} better than {i}")
          print(f"{texts[j]} THAN \n{texts[i]}")
  
    writer.write(utt)
    #break
      
  end = time.time()
  print(end-start, count)
  
  json.dump(data, open(model_name+'data_temp.json','w'), ensure_ascii=False, indent=2)