#!/usr/bin/env python3
import numpy as np
from hmmlearn import hmm
from utils import *
import sys
import pickle
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

use_saved_model = True
STATES = 52

target = sys.argv[1]

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

words = words[:1000]
sequences = np.array(
    [string_to_charlist(word) for word in words])
data = np.array([np.concatenate(sequences)]).T

len_list = [len(word) for word in words]

if not use_saved_model:
    model = hmm.GaussianHMM(
        n_components=STATES,
        n_iter=1000)

    model.fit(data, lengths=len_list)
    f = open('./saved_model', 'wb')
    f.write(pickle.dumps(model))
    f.close()
else:
    f = open('./saved_model', 'rb')
    model = pickle.loads(f.read())
    f.close()

prediction = model.predict(data)

decoder = {c: list() for c in range(STATES)}

for i, s in enumerate(prediction):
    alphabet = chr(int(data[i][0]) + ord('a') - 1)

    decoder[s].append(alphabet)


for k, v in decoder.items():
    if not v:
        continue
    decoder[k] = max(set(v), key=v.count)

print(decode_state(to_state(target, model), decoder))
