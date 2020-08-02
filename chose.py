from jiwer import wer
import jsonlines
import numpy as np

def strategy1_best(candidates, preds):
    return candidates[0]

def strategy2_oracle(candidates, preds):
    best_error=10000
    best_candidate=None
    for candidate in candidates:
        error = wer(obj['ref'], candidate)
        if error<best_error:
            best_error=error
            best_candidate=candidate
    return best_candidate

def gmean(input_x):
    log_x = np.log(input_x)
    return np.exp(np.mean(log_x))

def get_ranking(A):
  # "A linear model for ranking soccer teams"
  # macierz musi być nieujemna
  # dolna macierz, wyższe liczby oznaczają wygrane wierszy
  eigvalues, eigvectors = np.linalg.eig(A)
  #print(eigvalues, eigvectors)
  highest_eigvalue_index = np.argmax(eigvalues)
  eigvector = eigvectors[:,highest_eigvalue_index]
  # np.argmax(eigvector) # highest
  # TODO: wartosc bezwzgledna?
  print('eigvector', eigvector)
  return np.argmax(eigvector)

def strategy3_bert(candidates, preds):
    A=np.zeros((len(candidates),len(candidates))) 
    
    for i,rest in preds.items():
        for j, probs in rest.items():
            i=int(i)
            j=int(j)
            print(i, j, probs)
            score1 = sum(probs[0])
            score2 = sum(probs[1])
            #TODO: co z pustym ciągiem?
            
            #print('probs', probs)
            #score1 = np.mean(probs[0])
            #score2 = np.mean(probs[1])
            score1 = gmean(probs[0])
            score2 = gmean(probs[1])
            
            score=score1-score2
            #print(A, score, type(i), type(j))
            if score>0:
                print(f"{i} THAN {j}")
                A[i, j] = score
                #A[j, i] = -score
            else:
                print(f"{j} THAN {i}")
                A[j, i] = abs(score)
                #A[i, j] = -abs(score)

    #print(A)
    index = A.sum(axis=1).argmax() #największy wiersz
    print(A)
    index = get_ranking(A)
    print(index)
    return candidates[index]
    

strategy=strategy3_bert

errors=[]
path='xlm-roberta-baseoutput1_omg.jsonl'
with jsonlines.open(path) as reader:
    for obj in reader:
        #print(obj)
        prediction = strategy(obj["candidates"], obj["preds"])
        error = wer(obj['ref'], prediction)
        print(error)
        print(obj['ref'])
        print(prediction)
        
        errors.append(error)

print(errors)
print(np.mean(errors))

#s1: 0.13743334457528833
#s2: 0.04708791208791209
#s3: 0.08779634261731449

# czy token_id 6 oznacza nic?