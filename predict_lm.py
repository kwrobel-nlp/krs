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
    parser.add_argument('--nbest', help='path to nbest.txt', required=True)
    parser.add_argument('--output', help='Output path to JSONL', required=True)
    parser.add_argument('--reference', default=None, help='path to reference.txt')
    parser.add_argument('--nocuda', action='store_true', help='no CUDA')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--half', action='store_true', help='use model.half() (may cause token prob 0.0)')
    parser.add_argument('--opi', action='store_true', help='use OPI tokenizer')
    parser.add_argument('-c', default=None, help='continue using path to JSONL')
    args = parser.parse_args()

    device = 'cuda'
    if args.nocuda:
        device = 'cpu'

    print('CUDA', torch.cuda.is_available(), file=sys.stderr)

    model_name = args.model
    data = load_both(args.reference, args.nbest)

    processed_ids = set()
    if args.c is not None:
        with jsonlines.open(args.c) as reader:
            for obj in reader:
                processed_ids.add(obj['id'])

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

    start = time.time()
    count = 0
    with jsonlines.open(args.output, mode='w', flush=True) as writer:
        for id, utt in tqdm.tqdm(data.items(), desc="Texts"):
            if id in processed_ids: continue

            texts = utt['candidates']

            texts_ids = []
            for text in texts:
                if args.opi:
                    ids = torch.tensor([tokenizer.encode(text).ids])
                else:
                    ids = tokenizer.encode(text, return_tensors="pt")
                texts_ids.append(ids)

            utt['probas'] = []
            for i, text_ids in enumerate(tqdm.tqdm(texts_ids, desc="Cands")):
                logprob, length, logprob_wo0 = text_prob(text_ids, tokenizer.mask_token_id, model, device,
                                                         args.batch_size)
                utt['probas'].append((logprob, length, logprob_wo0))

            writer.write(utt)

        end = time.time()
        print('Time', end - start, count, file=sys.stderr)
