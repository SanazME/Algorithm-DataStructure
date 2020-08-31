"""
Given an array containing all the numbers from 1 to n except two, find the two missing numbers.

eg.

missing([4, 2, 3]) = 1, 5

1. is it an ordered array?
2. Always missing two elements?
3. are they integers?
"""

def missing(arr):
    num_elems=len(arr)+2
    sum_arr = num_elems * (num_elems+1)/2
    sum_missing = sum(arr)
    diff = sum_arr-sum_missing
    #print(diff)
    
    result=[]
    for i in range(num_elems):
        compl = diff-(i+1)
        if compl not in arr:
                result.append(compl)
    return result



print(missing([4,2,3]))