
source = open('train.fr').readlines()
with open('train.fr', 'w') as output:
    for line in source:
        output.write('\n')

source = open('valid.fr').readlines()
with open('valid.fr', 'w') as output:
    for line in source:
        output.write('\n')

source = open('test.fr').readlines()
with open('test.fr', 'w') as output:
    for line in source:
        output.write('\n')

