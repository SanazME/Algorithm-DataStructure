"""Given two integers N and M. The problem is to find the number
closest to N and divisible by M. If there are more than one such
number, then output the one having maximum absolute value.

Input:
The first line consists of an integer T i.e number of test cases.
T testcases follow.  The first and only line of each test case
contains two space separated integers N and M.

Output:
For each testcase, in a new line, print the closest number to N
which is also divisible by M.

Constraints:
1 <= T <= 100
-1000 <= N, M <= 1000

Example:
Input:
2
13 4
-15 6
Output:
12
-18

"""
def ClosestNumber(n,m):
    # Inner sign function
    sign = lambda x : 1 if (x>0) else -1 if (x<0) else 0

    if n%m==0:
        return n
    else:
        n_abs = abs(n)
        m_abs = abs(m)
        coeff = n_abs//m_abs
        num_less = coeff * m_abs
        num_more = (coeff+1)* m_abs
        if (num_more - n_abs) <= (n_abs-num_less):
            return num_more * sign(n)
        else:
            return num_less * sign(n)
test_cases = input()
results = []
for i in range(int(test_cases)):
    n,m = map(int, input().split())
    results.append(ClosestNumber(n,m))
for item in results:
    print(item)
