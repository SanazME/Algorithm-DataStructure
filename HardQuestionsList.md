## 1. Make Array Non-decreasing or Non-increasing
- https://leetcode.com/problems/make-array-non-decreasing-or-non-increasing/description/
- **Easy version:** https://leetcode.com/problems/non-decreasing-array/?envType=list&envId=o160a5j5

- **using Min-heap**
For non-increasing (decreasing) case:
- Calculate the sum of absolute differences between the final array elements and the current array elements. Thus, the answer will be the sum of the difference between the ith element and the smallest element that occurred until then.
- For this, we can maintain a min-heap to find the smallest element encountered till then. In the min-priority queue, we will put the elements, and new elements are compared with the previous minimum. If the new minimum is found we will update it, this is done because each of the next elements which are coming should be smaller than the current minimum element found till now. Here, we calculate the difference so that we can get how much we have to change the current number so that it will be equal or less than previous numbers encountered. Lastly, the sum of all these differences will be our answer as this will give the final value up to which we have to change the elements.
- so we paied the cost of changing the element, other that element or its previous one.
- For increasing case, it's the same but we use -1 so that we can use min heap again. Try to write for both cases the solution and then refactor the solution into one code:
```py
from queue import PriorityQueue

# for decreasing case
def convertArray2(nums):
    n = len(nums)
    if n <= 1:
        return 0

    # min heap
    pq = PriorityQueue()
    diff = 0

    # Here in the loop we will
    # check that whether the upcoming
    # element of array is less than top
    # of priority queue. If yes then we
    # calculate the difference. After
    # that we will remove that element
    # and push the current element in
    # queue. And the sum is incremented
    # by the value of difference
    for i in range(n):
        if not pq.empty():
            curr = pq.get()

            if curr < nums[i]:
                diff += nums[i] - curr
                pq.put(nums[i])
            else:
                pq.put(curr)

        pq.put(nums[i])
        
    return diff

# for increasing case
def convertArray2(nums):
    n = len(nums)
    if n <= 1:
        return 0

    pq = PriorityQueue()
    diff = 0

    for i in range(n):
        if not pq.empty():
            curr = pq.get()

            if -1 * curr > nums[i]:
                diff += nums[i] - abs(curr)
                pq.put(-nums[i])
            else:
                pq.put(curr)


        pq.put(-nums[i])
        
    return diff

# Combined
def convertArray(nums):
    n = len(nums)
    if n <= 1:
        return 0
    
    
    def helper(nums):
        pq = PriorityQueue()
        diff = 0
        
        for i in range(n):
            if not pq.empty():
                curr = pq.get()

                if curr < nums[i]:
                    diff += abs(nums[i] - curr)
                    pq.put(nums[i])
                else:
                    pq.put(curr)

            pq.put(nums[i])
        
        return diff
    
    return min(helper(nums), helper([-ele for ele in nums]))
```
