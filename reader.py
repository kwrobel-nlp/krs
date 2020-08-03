import collections
import sys

import regex


def preprocess_text(text):
    text = text.replace('<unk>', '')
    text = regex.sub('\s+', ' ', text)
    return text


def load(path, data):
    for line in open(path):
        id, text = line.split(' ', 1)
        text = text.strip()
        text = preprocess_text(text)
        ids = id.split('-')
        if len(ids) == 1:
            text_id = ids[0]
            data[text_id]['id'] = text_id
            data[text_id]['ref'] = text
            data[text_id]['type'] = 'sent' if 'sent' in text_id else 'rich'

        elif len(ids) == 2:
            text_id, pred_no = ids
            if 'id' not in data[text_id]: data[text_id]['id'] = text_id
            if 'type' not in data[text_id]: data[text_id]['type'] = 'sent' if 'sent' in text_id else 'rich'
            if 'candidates' not in data[text_id]:
                data[text_id]['candidates'] = []
            data[text_id]['candidates'].append(text)


def load_both(ref, nbest):
    data = collections.defaultdict(dict)
    if ref:
        load(ref, data)
    load(nbest, data)
    data = dict(data)
    return data
