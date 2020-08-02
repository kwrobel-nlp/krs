import json
import time
import jsonlines as jsonlines

from lm import text_prob
from reader import load_both
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import tqdm

device = 'cpu'

print(torch.cuda.is_available())

model_name = "xlm-roberta-large"

flatten = lambda l: [item for sublist in l for item in sublist]

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.eval()
if device == 'cuda':
    model.half()
    model.to(device)
    model.half()

text="oni nie są zdolni do rządzenia czarne to tak wypić epoce sport cały dzień nawiesza na siebie koralików i choć zadowolony pracować to nikt nie pracuje obniży jak sto lat temu"
text="oni nie są zdolne do rządzenia czarne to tak wypić a podczas par cały dzień nawiesza na siebie koralików i choć zadowolony pracować to nikt nie pracuje oni że jak sto lat temu"
text_ids = tokenizer.encode(text, return_tensors="pt")
logprob, length, logprob_wo0 = text_prob(text_ids, tokenizer.mask_token_id, model, device)
print(logprob, length, logprob_wo0)
