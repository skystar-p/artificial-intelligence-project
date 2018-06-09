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
TEST_FILE = 'test-1'

try:
    target = sys.argv[1]
except IndexError:
    target = ''

# load train dataset
file_name = './dataset/train-0'
words = load_file(file_name)

words = words[:1000]
sequences = np.array(
    [string_to_charlist(word) for word in words])
data = np.array([np.concatenate(sequences)]).T

len_list = [len(word) for word in words]

if not use_saved_model:
    # define gaussian hmm model
    model = hmm.GaussianHMM(
        n_components=STATES,
        n_iter=1000)

    # train with given sequences
    model.fit(data, lengths=len_list)
    f = open('./saved_model', 'wb')
    f.write(pickle.dumps(model))
    f.close()
else:
    f = open('./saved_model', 'rb')
    model = pickle.loads(f.read())
    f.close()

# find most likely path by viterbi algorithm
prediction = model.predict(data)

# interpret states
decoder = {c: list() for c in range(STATES)}

for i, s in enumerate(prediction):
    alphabet = chr(int(data[i][0]) + ord('a') - 1)

    decoder[s].append(alphabet)

# decode state sequence to alphabet sequence
for k, v in decoder.items():
    if not v:
        continue
    decoder[k] = max(set(v), key=v.count)

# result alphabet sequence
if target:
    print(decode_state(to_state(target, model), decoder))

# test
file_name = './dataset/' + TEST_FILE
words = load_file(file_name)
print('Wordcount: {}'.format(len(words)))
sequences = np.array(
    [string_to_charlist(word) for word in words])
data = np.array([np.concatenate(sequences)]).T

prediction = model.predict(data)

decoder = {c: list() for c in range(STATES)}

for i, s in enumerate(prediction):
    alphabet = chr(int(data[i][0]) + ord('a') - 1)

    decoder[s].append(alphabet)

for k, v in decoder.items():
    if not v:
        continue
    decoder[k] = max(set(v), key=v.count)

total_corrected = 0
for word in words:
    decoded = decode_state(to_state(word, model), decoder)
    if word == decoded:
        total_corrected += 1

accuracy = total_corrected / len(words) * 100
print('Bypass accuracy: {}%'.format(accuracy))

# typo correction accuracy
file_name = './dataset/{}-typo'.format(TEST_FILE)
typo_words = load_file(file_name)

total_corrected = 0
for i, word in enumerate(typo_words):
    decoded = decode_state(to_state(word, model), decoder)
    if words[i] == decoded:
        total_corrected += 1

accuracy = total_corrected / len(words) * 100
print('Typo correction accuracy: {}%'.format(accuracy))
