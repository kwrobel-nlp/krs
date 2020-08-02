# KRS

Count probability of a text using transformer MLM model.

Data available at: http://poleval.pl/tasks/task1/

## Rescore

### 1. Copy `nbest.txt` to `data/test2`

### 2. Calculate probabilities
```
python3 predict_lm.py --model xlm-roberta-base --dir data/test2 --output xlm-roberta-base_test2_lm.jsonl
```
For PolishRoberta argument `--opi` is needed.

### 3. Choose best
```
python3 chose_lm.py xlm-roberta-base_test2_lm.jsonl 2>/dev/null > predictions.txt
```

## Results

|                                  |    dev |   test |  test2 |
|----------------------------------|-------:|-------:|-------:|
| oracle from 100-best             |  6.13% |  5.92% |        |
| 1-best                           | 12.09% | 12.22% |        |
| xlm-roberta-large                |  9.25% |  8.86% |        |
| xlm-roberta-large ppl            |  9.68% |  9.44% |        |
| xlm-roberta-large + preprocess   |  8.53% |  8.14% |        |
| polishroberta-large + preprocess |  **7.77%** |  **7.81%** |        |
| polishroberta-base + preprocess  |  8.15% |  7.99% |        |