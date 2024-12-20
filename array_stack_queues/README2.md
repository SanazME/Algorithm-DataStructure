## Top K Frequent Elements
- https://leetcode.com/problems/top-k-frequent-elements/description
- for getting a better time complexity than O(n log n), we can use heap to save the most k frequent elements. Basically, we iterate through the list and add elements to the heap with their freq and if an element freq is larger than the heap first ele, we then pop that element and replace it. At the end, we will have the most k frequent elements
```py
from heapq import *
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        if not nums:
            return None
        
        if len(nums) == 1:
            return nums
        
        freq_map = Counter(nums)
        
        heap = []
        
        # create a heap of size k
        for num, freq in freq_map.items():
            if len(heap) < k:
                heappush(heap, (freq, num))
            else:
                if heap[0][0] < freq:
                    heappop(heap)
                    heappush(heap, (freq, num))
                    
                    
        return [num for _, num in heap]
```
