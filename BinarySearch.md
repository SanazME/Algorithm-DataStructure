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

- also in an ascending array `[3,5,8,8,8,9,10]`, to find the first occurance of target (`8`) is to use the above template:
```py
def search(nums, target):
    left, right = 0, len(nums)
    
    while left < right:
        mid  = (left + right) // 2
        
        if nums[mid] >= target:
            right = mid
        else:
            left = mid + 1
    return left
```

**2. template 2:**
```py
def binarySearch(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    if len(nums) == 0:
        return -1

    left, right = 0, len(nums)
    
    while left < right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid

    # Post-processing:
    # End Condition: left == right
    if left != len(nums) and nums[left] == target:
        return left
    return -1

 ```
 **Distinguishing Syntax:**
  - Initial Condition: `left = 0, right = length`
  - Termination: `left == right`
  - Searching Left: `right = mid`
  - Searching Right: `left = mid+1`

-**First bad version**: https://leetcode.com/problems/first-bad-version/

- **Find min in ratated sorted array; ***: https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/
- If the array is rotated, there will be an inflection point, where all points on the left of it are larger than the first element of the rotated array and all elements on its right are smaller than the first element.
```py
def findMin(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    if nums[0] < nums[-1]:
        return nums[0]

    left, right = 0, len(nums) - 1
    min_sofar = float('Inf')

    while left < right:
        mid = (left + right) // 2
        # inflection point is on the left
        if nums[mid] < nums[0]:
            right = mid

        elif nums[mid] > nums[-1]:
            left = mid + 1

    return nums[left]
```

**3. template 3:**
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
    while left + 1 <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid
        else:
            right = mid

    # Post-processing:
    # End Condition: left + 1 == right
    if nums[left] == target: return left
    if nums[right] == target: return right
    return -1
 ```
 -  It is used to search for an element or condition which requires accessing the current index and its immediate left and right neighbor's index in the array.
 
 **Distinguishing Syntax:**
  - Initial Condition: `left = 0, right = length-1`
  - Termination: `left + 1 = right`
  - Searching Left: `right = mid`
  - Searching Right: `left = mid`
