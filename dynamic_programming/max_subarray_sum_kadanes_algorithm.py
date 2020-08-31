"""
Given an array, find the subarray(contingous elements) with max sum:
"""

def max_sum_subarray2(arr):

    max_sofar = arr[0]
    max_sum = arr[0]

    for i in range(1,len(arr)):
        max_sofar = max(arr[i] , arr[i] + max_sofar)
        max_sum = max(max_sum, max_sofar)
    return max_sum

def max_sum_subarray(arr):

    if len(arr) == 0:
        return 0
    else:
        max_sofar = arr[0]
        max_sum = arr[0]

        for item in arr:
            max_sofar = max(0 , item + max_sofar)
            max_sum = max(max_sum, max_sofar)
        return max_sum

max_sum_subarray2([-2, 1, -3, 4, -1, 2, 1, -5, 4])
