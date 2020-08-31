"""
Problem Statement:

You are given a read-only array of n integers from 1 to n. Each integer appears exactly once except A which appears twice and B which is missing.

Return A and B.

Example

Input:[3 1 2 5 3] 
Output:[3, 4] 
A = 3, B = 4
"""

def missingAndDuplicates(nums):
    original_nums = [x for x in range(min(nums), max(nums)+1)]
    missing = list(set(nums)^set(original_nums))
    
    dups=set()
    occurance={}
    for item in nums:
        if item in occurance:
            occurance[item]+=1
            dups.add(item)
        else:
            occurance[item]=1

    return [missing, list(dups)]

print(missingAndDuplicates([3,1,2,5,3,2,7,2]))
print(missingAndDuplicates([1, 2, 3, 4, 6]))
