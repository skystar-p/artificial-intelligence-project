import numpy as np
from hmmlearn import hmm
from utils import string_to_charlist


# load train dataset
file_name = './dataset/train-0'
f = open(file_name, 'r')
content = f.read()
f.close()

raw_words = content.split('_')

words = []
for w in raw_words:
    if w.isalpha():
        words.append(w)

sequences = np.array(
    [string_to_charlist(word) for word in words])

len_list = [len(word) for word in words]

model = hmm.GaussianHMM(
    n_components=26,
    n_iter=1000)

data = np.array([np.concatenate(sequences)])
# data = np.concatenate([['a', 'p', 'p', 'l', 'e'], ['o', 'r', 'a', 'n', 'g', 'e']])
# data = [ord(c) for c in data]
data = np.array(data).T
print(data)
model.fit(data, lengths=len_list)
