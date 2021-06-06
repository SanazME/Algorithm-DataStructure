## Binary Search
- **3 parts of a Binary Search**:
Binary Search is generally composed of 3 main sections:

1. Pre-processing - Sort if collection is unsorted.
2. Binary Search - Using a loop or recursion to divide search space in half after each comparison.
3. Post-processing - Determine viable candidates in the remaining space.

### 3 Templates
**1. template 1:**
```py
def binarySearch(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    if len(nums) == 0:
        return -1

    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    # End Condition: left > right
    return -1
 ```
 **Distinguishing Syntax:**
  - Initial Condition: `left = 0, right = length-1`
  - Termination: `left > right`
  - Searching Left: `right = mid-1`
  - Searching Right: `left = mid+1`
    
    
 - **Search in Rotated Sorted Array:**
 - https://leetcode.com/problems/search-in-rotated-sorted-array/
 - Explanation: Let's say nums looks like this: [12, 13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]. Because it's not fully sorted, we can't do normal binary search. But here comes the trick:
    - If target is let's say 14, then we adjust nums to this, where "inf" means infinity: [12, 13, 14, 15, 16, 17, 18, 19, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf]

    - If target is let's say 7, then we adjust nums to this: [-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
And then we can simply do ordinary binary search. 
Of course we don't actually adjust the whole array but instead adjust only on the fly only the elements we look at. And the adjustment is done by comparing both the target and the actual element against nums[0].
```py
def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        left, right = 0, len(nums)
        
        while left < right:
            mid = (left + right) // 2
            
            # both target and mid are on the same side
            if (target < nums[0]) == (nums[mid] < nums[0]):
                num = nums[mid]
            elif target < nums[0]: # target is on the right side
                num = -float('Inf')
            else:
                num = float('Inf')
               
            print(num)
            if num < target:
                left = mid + 1
            elif num > target:
                right = mid
            else:
                return mid
        return -1
```

