import sys

from argparse import ArgumentParser

from lm import correct_spaces

from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import tqdm

if __name__ == '__main__':
    parser = ArgumentParser(description='Transformer MLM as LM')
    parser.add_argument('--model', default='xlm-roberta-large', help='model', required=True)
    parser.add_argument('--path', help='path to predictions TXT', required=True)
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

    from tokenizers import SentencePieceBPETokenizer
    SentencePieceBPETokenizer

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

    lines = open(args.path).readlines()
    for line in tqdm.tqdm(lines):
        id, text = line.split(' ', 1)

        if args.opi:
            ids = torch.tensor([tokenizer.encode(text).ids])
        else:
            ids = tokenizer.encode(text, return_tensors="pt")

        new_text = correct_spaces(ids, tokenizer, model, device, args.batch_size)
        print(id, new_text)
