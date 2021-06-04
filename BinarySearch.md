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
    
