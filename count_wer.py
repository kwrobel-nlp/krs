from argparse import ArgumentParser

from jiwer import wer
import numpy as np

if __name__ == '__main__':
    parser = ArgumentParser(description='Count WER')
    parser.add_argument('ref', help='reference path')
    parser.add_argument('pred', help='predictions path')
    args = parser.parse_args()

    errors = []
    for line_ref, line_pred in zip(open(args.ref), open(args.pred)):
        id_ref, text_ref = line_ref.split(' ', 1)
        id_pred, text_pred = line_pred.split(' ', 1)
        assert id_ref == id_pred
        error = wer(text_ref, text_pred)
        errors.append(error)
    print(np.mean(errors))
