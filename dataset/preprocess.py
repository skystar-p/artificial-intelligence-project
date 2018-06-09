import sys
import re


filename = sys.argv[1]

f = open(filename, 'rb')
content = f.read()
f.close()

new_content = content.decode()
removal = '-,.;:\"/?[]{}=+\'*!@#$%^&()'

for c in list(removal):
    print('Replacing {}'.format(c))
    new_content = new_content.replace(c, '')

new_content = new_content.lower()
new_content = re.sub('\s+', '_', new_content)
new_content = new_content.strip('_')

f = open(filename, 'w')
f.write(new_content)
f.close()
