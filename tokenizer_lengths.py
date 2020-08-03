import sys
import time
from argparse import ArgumentParser

import jsonlines as jsonlines

from lm import text_prob
from reader import load_both
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import tqdm
import numpy as np

if __name__ == '__main__':
    parser = ArgumentParser(description='Transformer MLM as LM')
    parser.add_argument('--model', default='xlm-roberta-large', help='model', required=True)
    parser.add_argument('--dir', help='directory with reference.txt and nbest.txt', required=True)
    parser.add_argument('--opi', action='store_true', help='use OPI tokenizer')
    parser.add_argument('-c', default=None, help='continue using path to JSONL')
    args = parser.parse_args()

    model_name = args.model
    directory = args.dir
    data = load_both(f'{directory}/reference.txt', f'{directory}/nbest.txt')

    processed_ids = set()
    if args.c is not None:
        with jsonlines.open(args.c) as reader:
            for obj in reader:
                processed_ids.add(obj['id'])

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if args.opi:
        from tokenizers import SentencePieceBPETokenizer
        from tokenizers.processors import RobertaProcessing

        tokenizer = SentencePieceBPETokenizer(f"{args.model}/vocab.json", f"{args.model}/merges.txt")
        getattr(tokenizer, "_tokenizer").post_processor = RobertaProcessing(sep=("</s>", 2), cls=("<s>", 0))

    sizes = []
    for id, utt in tqdm.tqdm(data.items(), desc="Texts"):
        if id in processed_ids: continue

        texts = utt['candidates']

        
        for text in texts:
            if args.opi:
                ids = torch.tensor([tokenizer.encode(text).ids])
            else:
                ids = tokenizer.encode(text, return_tensors="pt")

            sizes.append(ids.size(1))

    print(np.mean(sizes), np.max(sizes))
