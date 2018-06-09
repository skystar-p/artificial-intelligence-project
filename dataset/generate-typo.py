import random
import sys


filename = sys.argv[1]

f = open(filename, 'rb')
content = f.read().decode()
f.close()

words = content.split('_')

new_words = []

for word in words:
    l = random.randint(0, len(word) - 1)
    ra = chr(random.randint(ord('a'), ord('z')))
    new_word = word[:l] + ra + word[l + 1:]
    new_words.append(new_word)

f = open(filename + '-typo', 'w')
f.write('_'.join(new_words))
f.close()
