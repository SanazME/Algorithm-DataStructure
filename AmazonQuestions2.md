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
