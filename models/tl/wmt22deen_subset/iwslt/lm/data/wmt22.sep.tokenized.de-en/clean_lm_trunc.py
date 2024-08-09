
source = open('train.de').readlines()
with open('train.de', 'w') as output:
    for line in source:
        output.write('\n')

source = open('valid.de').readlines()
with open('valid.de', 'w') as output:
    for line in source:
        output.write('\n')

source = open('test.de').readlines()
with open('test.de', 'w') as output:
    for line in source:
        output.write('\n')

