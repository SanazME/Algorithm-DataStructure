'''You are given 2 numbers (N , M); the task is to
find N√M (Nth root of M).

Input:
The first line of input contains an integer T
denoting the number of test cases. Then T test cases
follow. Each test case contains two space separated
integers N and M.

Output:
For each test case, in a new line, print an integer
denoting Nth root of M if the root is an integer
else print -1.

Constraints:
1 <= T <= 200
2 <= N <= 20
1 <= M <= 109+5

Example:
Input:
2
2 9
3 9
Output:
3
-1'''
import math
def NthRootNumber(n,m):
    log10_root = (math.log10(m))/n
    floor = math.floor(pow(10, log10_root))
    ceil = math.ceil(pow(10, log10_root))
    if pow(floor, n)==m:
        return floor
    elif pow(ceil, n)==m:
        return ceil
    else:
        return -1





# Older method does not work for larger numbers m = 10^9
from decimal import Decimal, getcontext

def NewtonsMethod(n, A, precision):
    getcontext().prec = precision

    n = Decimal(n)
    x_0 = A / n #step 1: make a while guess.
    x_1 = 1     #need it to exist before step 2
    while True:
        #step 2:
        x_0, x_1 = x_1, (1 / n)*((n - 1)*x_0 + (A / (x_0 ** (n - 1))))
        if x_0 == x_1:
            return x_1

def NthRoot(n,m):
    root = NewtonsMethod(n,m, 6)
    if root == int(root):
        return int(root)
    else:
        return -1
test_cases = input()
results = []
for i in range(int(test_cases)):
    n,m = map(int, input().split())
    results.append(NthRoot(n,m))

for item in results:
    print(item)
