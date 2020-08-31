# Uses python3
# There are two ways of running this program:
# 1. Run
#     python3 APlusB.py
# then enter two numbers and press ctrl-d/ctrl-z
# 2. Save two numbers to a file -- say, dataset.txt.
# Then run
#     python3 APlusB.py < dataset.txt

import sys

print('Enter two integers:')
input = sys.stdin.read()
try:
    tokens = input.split()
    a = int(tokens[0])
    b = int(tokens[1])
    print(a + b)
except:
    print('No cool!') 

