"""
Write a Python program to find missing numbers from a list.
The missing numbers can be more than 1
"""

def findMissingNumbers(nums):
    """
    input: nums - list[integrs]
    outout: list[integers]
    """

    original_nums=[x for x in range(nums[0], nums[-1]+1)]
    nums_set = set(nums)

    # XOR: Exclusive or. It is one when only one is true.
    return list(nums_set ^ set(original_nums))


print(findMissingNumbers([10,11,12,14,17]))