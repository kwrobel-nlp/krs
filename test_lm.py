import sys
import time
from argparse import ArgumentParser

import jsonlines as jsonlines

from lm import text_prob
from reader import load_both
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import tqdm

if __name__ == '__main__':
    parser = ArgumentParser(description='Transformer MLM as LM')
    parser.add_argument('--model', default='xlm-roberta-large', help='model', required=True)
    parser.add_argument('--text', help='text to process', required=True)
    parser.add_argument('--nocuda', action='store_true', help='no CUDA')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--half', action='store_true', help='use model.half() (may cause token prob 0.0)')
    parser.add_argument('--opi', action='store_true', help='use OPI tokenizer')
    args = parser.parse_args()

    device = 'cuda'
    if args.nocuda:
        device = 'cpu'

    print('CUDA', torch.cuda.is_available(), file=sys.stderr)

    model_name = args.model

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    if args.opi:
        from tokenizers import SentencePieceBPETokenizer
        from tokenizers.processors import RobertaProcessing

        tokenizer = SentencePieceBPETokenizer(f"{args.model}/vocab.json", f"{args.model}/merges.txt")
        getattr(tokenizer, "_tokenizer").post_processor = RobertaProcessing(sep=("</s>", 2), cls=("<s>", 0))
        tokenizer.mask_token_id = model.roberta.embeddings.word_embeddings.weight.shape[0] - 1  # last is mask?

    model.eval()

    if device == 'cuda':
        if args.half:
            model.half()
        model.to(device)

    text = args.text

    if args.opi:
        ids = torch.tensor([tokenizer.encode(text).ids])
    else:
        ids = tokenizer.encode(text, return_tensors="pt")

    logprob, length, logprob_wo0 = text_prob(ids, tokenizer.mask_token_id, model, device, args.batch_size)
    print(logprob, length)
