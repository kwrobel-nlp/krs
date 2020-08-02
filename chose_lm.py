import sys
from argparse import ArgumentParser

from jiwer import wer
import jsonlines
import numpy as np


def strategy1_best(candidates, logprobas):
    return candidates[0]


def strategy2_oracle(candidates, logprobas):
    best_error = 10000
    best_candidate = None
    for candidate in candidates:
        error = wer(obj['ref'], candidate)
        if error < best_error:
            best_error = error
            best_candidate = candidate
    return best_candidate


def gmean(input_x):
    log_x = np.log(input_x)
    return np.exp(np.mean(log_x))


def get_ranking(A):
    # "A linear model for ranking soccer teams"
    # macierz musi być nieujemna
    # dolna macierz, wyższe liczby oznaczają wygrane wierszy
    eigvalues, eigvectors = np.linalg.eig(A)
    highest_eigvalue_index = np.argmax(eigvalues)
    eigvector = eigvectors[:, highest_eigvalue_index]
    # np.argmax(eigvector) # highest
    return np.argmax(eigvector)


def strategy3_bert(candidates, logprobas):
    best_score = None
    best_candidate = None
    for logproba, candidate in zip(logprobas, candidates):
        logprob, length, logprob_wo0 = logproba
        prob = np.exp(logprob)
        ppl = np.exp(-logprob / length)
        prob_wo0 = np.exp(logprob_wo0)
        print(logprob, length, prob, ppl, prob_wo0, candidate, file=sys.stderr)
        # dla -inf prob to 0 a ppl inf
        score = prob
        if best_score is None or score > best_score:
            best_score = score
            best_candidate = candidate
    return best_candidate


if __name__ == '__main__':
    parser = ArgumentParser(description='Choose best candidate and calculate WER')
    parser.add_argument('path', help='Path to JSONL')
    args = parser.parse_args()

    strategy = strategy3_bert

    errors = []
    path = args.path
    with jsonlines.open(path) as reader:
        for obj in reader:
            prediction = strategy(obj["candidates"], obj["probas"])
            if 'ref' in obj:
                error = wer(obj['ref'], prediction.replace('<unk>', ''))
                errors.append(error)
                print(error, file=sys.stderr)
                print(obj['ref'], file=sys.stderr)
            print(obj['id'], prediction)

    if errors: print('WER', np.mean(errors), file=sys.stderr)
