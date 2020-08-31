"""
LCS : Longest Common Subsequent

Total number of subsequents in a sequence (or a set):

sigma k=0->n C(n,k) = 2^n (binomial factorial)

https://www.techiedelight.com/longest-common-subsequence/


"""

def find_LCS(X,Y):
    """
    find the largest common subsequence of 2 sequences
    """

    n=len(X)
    m = len(Y)

    if n==0 or m==0:
        return 0
    else:
        # Last elements are identical
        if X[-1]==Y[-1]:
            return find_LCS(X[:-1],Y[:-1])+1
        else:
            # last elements are not identical
            return max(find_LCS(X[:-1],Y), find_LCS(X,Y[:-1]))


X='ABCBDAB'
Y = 'BDCABA'
find_LCS(X,Y)

"""
The worst runtime is O(2^(n+m)) when there is no common subsequence. The memoization
makes the time complexity down to O(nm). The space complexity of below fun is also
O(nm)

Difference between top-down and bottom-up approaches in dynamic programming:
https://www.techiedelight.com/introduction-dynamic-programming/?v=1#top-down

Dynamic programming:
1.optimal substructure
2. overlapping sub-problems:
    2.1. top-down approach, memoization (using a map or array)
    2.2  bottom-up approach, tabulation

If a problem can be solved by combining optimal solutions to non-overlapping sub-problems
,the stategy is called "divide and conquer". Merge sort and quick sort are not dynamic_programming.


"""


def find_LCS_dyn(X,Y,memory={}):

    n = len(X)
    m = len(Y)

    if n==0 or m==0:
        return 0

    key = str(n)+"|"+str(m)

    if key not in memory.keys():
        if X[-1]==Y[-1]:
            memory[key] = find_LCS_dyn(X[:-1],Y[:-1], memory)+1
        else:
            memory[key] = max(find_LCS_dyn(X[:-1], Y, memory), find_LCS_dyn(X, Y[:-1], memory))

    return memory[key]

X='XMJYAUZ'
Y = 'MZJAWXU'
find_LCS_dyn(X,Y)

"""
Tabulated approach instead of memoization
"""
def find_LCS_bottom_up(X,Y):

    n=len(X)
    m=len(Y)

    """
    lookup table stores solution to already computed subproblems. table[i][j] stores
    the lenght of LCS of substring X[0:i-1] Y[0:j-1]
    """
    table = [[None]*(m+1) for i in range(n+1)]
    print(table)
    for i in range(n+1):
        for j in range(m+1):
            # If either X or Y has zero length
            if i==0 or j==0:
                table[i][j] = 0
            # If the last elements are the same
            elif X[i-1]==Y[j-1]:
                table[i][j] = table[i-1][j-1]+1
            else:
                table[i][j] = max(table[i-1][j], table[i][j-1])

    return table
X='XMJYAUZ'
Y = 'MZJAWXU'
find_LCS_bottom_up(X,Y)


"""
Improve the space complexity from O(nm) to O(n) in bottom-top approach:
For each row, we need the reesults of all columns in the previous row so instead of
keeping track of all rows, we just keep track of the previous row and the current row.
"""

def find_LCS_table_space(X,Y):
    n = len(X)
    m = len(Y)

    #Create a table with 2 rows only
    table = [[None]*(m+1)]*2

    for i in range(n+1):
        binary_i = i and 1
        for j in range(m+1):
            if binary_i==0 or j==0:
                table[binary_i][j]=0

            elif X[i-1]==Y[j-1]:
                # The last characters are the same in both strings
                table[binary_i][j] = table[binary_i-1][j-1] + 1

            else:
                table[binary_i][j] = max(table[binary_i-1][j], table[binary_i][j-1])

    return table[binary_i][j]

    X = 'XMJYAUZ'
    Y = 'MZJAWXU'
    find_LCS_table_space(X,Y)
