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

## to find the first and last occurance of a number in an array
- we can use binary search twice, once for finding the first occurance (the position) and the second for the position of the last occurance of that number. The runtime would be **O(logn)**.
- To find the first occurance, we use template 2 to check with the right hand-side element of the midpoint.
- To find the last occurance, we use template 3 to check left and right hand side of the midpoint.
- https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/
```py
def searchRange(nums, target):
    if not nums:
        return [-1, -1]
        
    def start(n):
        left, right = 0, len(nums) - 1
        
        while left < right:
            mid = (left + right) // 2
            
            if nums[mid] >= n:
                right = mid
            else:
                left = mid + 1
        if nums[left] == n:
            return left
        else:
            return -1
    
    
    def finish(n):
        left, right = 0, len(nums) - 1
        
        while left + 1 < right:
            mid = (left + right)//2

            if nums[mid] <= n:
                left = mid
            else: 
                right = mid
        if nums[right] == n:
            return right
        elif nums[left] == n:
            return left
        else:
            return -1
    
    start = start(target)
    end = finish(target)
    
    return [start, end]
```

## Median of Two Sorted Arrays (https://leetcode.com/problems/median-of-two-sorted-arrays/)
- In O(n+m) runtime:
```py
def findMedianSortedArrays(nums1, nums2):
    """
    :type nums1: List[int]
    :type nums2: List[int]
    :rtype: float
    """
    totalSize = len(nums1) + len(nums2)
        
    # if totalSize is even, we take result[totalSize-1] & resutl[totalSize]
    midLocation = totalSize // 2

    i1, i2 = 0, 0
    r = 0
    result = []

    while r <= midLocation:
        if i1 < len(nums1) and i2 < len(nums2):
            if nums1[i1] < nums2[i2]:
                result.append(nums1[i1])
                i1 += 1
            else:
                result.append(nums2[i2])
                i2 += 1
        else:
            if i1 < len(nums1):
                result.append(nums1[i1])
                i1 += 1
            if i2 < len(nums2):
                result.append(nums2[i2])
                i2 += 1

        r += 1

    if totalSize % 2 == 0:
        return float(result[-1] + result[-2])/2
    else:
        return result[-1]
```



- In O(log(min(n,m))) runtime (https://www.youtube.com/watch?v=LPFhl65R7ww)
```py
def findMedianSortedArrays(nums1, nums2):
    """
    :type nums1: List[int]
    :type nums2: List[int]
    :rtype: float
    """
    totalSize = len(nums1) + len(nums2)
    isTotalSizeEven = True if totalSize % 2 == 0 else False
    
    """
    Conditions for finding a median in two sorted arrays with length X, Y:
    1. if 
        partionX: number of elements left of the partition line
        partitionY: number of elements left of the partition line
        
        partitionX + partitionY = (X + Y + 1) / 2
        => length of the overall left partition == length of the overall right partition
    2. if 
        ..., maxLeftX | minRightX, ....
        ..., maxLeftX | minRightY, ...
        
        if maxLeftX <= minRightY & maxLeftY <= minRightX =>
            for Even number in total array: 
                median = avg(max(maxLeftX, maxLeftY), min(minRightX, minRightY))
            else:
                median = max(maxLeftX, maxLeftY)
                
        elif maxLeftX > minRightY, we need to move to left of X
        else: we need to move to right of X
        
        for part 2, we use binary search to search the location of the partition in  shortest array and then from that, the location of the partition on the other array will be derived.
    """
    sizeX = len(nums1)
    sizeY = len(nums2)
    
    # start with the shortest array
    if sizeY < sizeX:
        return findMedianSortedArrays(nums2, nums1)
    
    low, high = 0, sizeX
    
    while low <= high:
        
        partitionX = (low + high + 1) / 2
        partitionY = (sizeX + sizeY + 1)/2 - partitionX
        
        print(low, high, partitionX, partitionY)
        maxLeftX = float('-Inf') if partitionX == 0 else nums1[partitionX-1]
        minRightX = float('Inf') if partitionX == sizeX else nums1[partitionX]
        
        maxLeftY = float('-Inf') if partitionY == 0 else nums2[partitionY-1]
        minRightY = float('Inf') if partitionY == sizeY else nums2[partitionY]
        
        print(maxLeftX,minRightY, maxLeftY, minRightX)
        
        if maxLeftX <= minRightY and maxLeftY <= minRightX:
            if isTotalSizeEven:
                median = float(max(maxLeftX, maxLeftY)+ min(minRightX, minRightY)) / 2
            else:
                median = max(maxLeftX, maxLeftY)
            
            return median

        
        elif maxLeftX > minRightY:
            high = partitionX - 1
        else:
            low = partitionX + 1
```

## Find the missing number in Arithmetic progression
- The O(n) time solution is to iterate throught the array and find the missing number knowing that **if there is only one element is missing, we can find the difference of arthimetic progression from `arr[-1] - arr[0] / len(arr)` because with that missing element we would have (x+nk - x)/n+1 **
```py
def missingNum(arr):
    delta = (arr[0] + arr[-1])/len(arr)
    
    for i in range(1, len(arr)):
        localDiff = arr[i] - arr[i-1]
        
        if localDiff != delta:
            return arr[i-1] + delta
```

- To find it in O(logn), since the array is in order, we can use binary search.
