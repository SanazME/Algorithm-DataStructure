## 30. Find all contiguous subarrays that sum up to the given n value
- https://www.educative.io/courses/decode-coding-interview-python/xopDqXlkGrq
- We can create cumulative sum at each indices and have a hashTable to save the cum sum as a key and its occurance as value. Then if the difference between two cumSum at indices `cumSum[i] - cumSum[j] = n` it means that the sum of numbers in between these two indices is n.

```py
import collections
def allocateSpace(processes, n):
    cumSum = collections.defaultdict(int)
    cumSum[0] = 1
    count = 0
    totalSum = 0
    
    for process in processes:
        totalSum += process
        
        print('totalSum - n: ', totalSum - n)
        if totalSum - n in cumSum:
            count += cumSum[totalSum - n]
            
        cumSum[totalSum] += 1
        print(cumSum)
        
    return count
```
## 31. Resume nth preempted process in the nth round of process resumption
- https://www.educative.io/courses/decode-coding-interview-python/RLP064XLg8R
- We're finding a nth missing number is a sorted array of processes IDs. We use binary search since it's sorted array.
- recursice and iterative solutions: 
```py
def resumeProcess(arr, n):
    
    left, right = 0, len(arr) - 1
    
    while left + 1 < right:
        mid = (left + right) // 2
        missing = arr[mid] - arr[left] - (mid - left)
        
        if n > missing:
            # on the right half
            left = mid
            n -= missing
        else:
            right = mid

        
    return arr[left] + n
    
# recursive
def resumeProcess(arr, n):
    
    def helper(left, right, n):
        # base case
        if left + 1 == right:
            return arr[left] + n
        
        mid = (left + right) // 2
        missing = arr[mid] - arr[left] - (mid - left)
        if n > missing:
            # right half
            return helper(mid, right, n - missing)
        else:
            return helper(left, mid, n)
    
    
    pid = helper(0, len(arr) - 1, n)
    
    return pid

```
