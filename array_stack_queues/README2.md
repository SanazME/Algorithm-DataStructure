## Comparison in Python
The comparison uses lexicographical ordering: first the first two items are compared, and if they differ this determines the outcome of the comparison; if they are equal, the next two items are compared, and so on,
```py
[1, 2, 3]              < [1, 2, 4]
'ABC' < 'C' < 'Pascal' < 'Python'
```
## Recall that Bucket Sort is the sorting algorithm where items are placed at Array indexes based on their values (the indexes are called "buckets")

### Top K Frequent Elements
- https://leetcode.com/problems/top-k-frequent-elements/description
- If we don't need to return in any other ordering other than the most frequent and return the results **in any order**, we can just use min heap with size of `k` to filter out smallest counts. In this case, it's also guaranteed that the answer is **unique**. Meaning that we won't have multipe entries with the same count that we need to decide and order further based on other criteria.
-  However this approach won't work if in addition to frequncy we also need to return results in order of index or lexicographically or etc. In that case, take a look at the next problem.
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
### Top k Frequent words
- https://leetcode.com/problems/top-k-frequent-words/description/
- here we can't just filter out lowest counts of words in min heap because we need to decide for the same freq, the words with min lexicographical order will be before other words. For this case, we dont limit the length of our min heap and also try to use it as max heap so that at the end we can remove the first k words from heap. This works if the input size is small otherwise we end up creating a very large heap.
```py
def topKFrequent(self, words: List[str], k: int) -> List[str]:
        if k == 0:
            return []
        if len(words) <= 1:
            return words
        
        freq = Counter(words)
        heap = []
        
        for word, count in freq.items():
            heappush(heap, (-count, word))
        
        i = 0
        result = []
        while i < k:
            _, word = heappop(heap)
            result.append(word)
            i += 1

        return result
```
### Kth Largest Element in an Array
- https://leetcode.com/problems/kth-largest-element-in-an-array/description/
- Here we can use min heap to store the number and we don't need any extra comparable value, just the number itself. If the lenght of heap exceeds we pop it. Note that in the heap we have repeate numbers. Because it is the min-heap the top of the heap **will be the kth largest element**:
```py
class Solution:
    def findKthLargest(self, nums, k):
        nums.sort(reverse=True)
        return nums[k - 1]
```

### Given two arrays/lists, write a function that merges them up to the nth item. For example, if n=2:
List1: ['a', 'b', 'c']
List2: [1, 2, 3]
Result should merge only first 2 items from each list

- we can use slice to values:
```py
def merge(list1, list2, n):
    if not list1:
        return list2[:n]
    if not list2:
        return list1[:n]
    
    result = []
    for i in range(n):
        if i < len(list1):
            result.append(list1[i])
        if i < len(list2):
            result.append(list2[i])
    return result
```

### 1772. Sort Features by popularity
- https://leetcode.com/problems/sort-features-by-popularity/description/
```py
def sortFeatures(self, features: List[str], responses: List[str]) -> List[str]:
        freq = {}
        for rep in responses:
            words = set(rep.split(' '))

            for feat in features:
                if feat in words:
                    freq[feat] = freq.get(feat, 0) + 1

        return sorted(features, key = lambda x: freq.get(x, 0), reverse=True)   

```

### 451 Sort Characters by their frequencies
- https://leetcode.com/problems/sort-characters-by-frequency/editorial/
**Solution 1**
  - use hashmap and then heap to get the most frequent letters: **Time : O(n log n) and space O(n)**
```py
def frequencySort(self, s: str) -> str:
    if len(s) <= 1:
        return s
    
    freq = Counter(s)

    heap = []

    for char, count in freq.items():
        heappush(heap, (-count, char))

    result = []
    while heap:
        count, char = heappop(heap)
        result.extend([char for _ in range(-1*count)])

    return ''.join(result)
```
**Solution 2**:
- **for Time complexity O(n)** we can use **bucket sort** for a string of size n, max frequency of a char is `n` so we can have an array of size n and save the char at each index where index represent the frequency of that char.  We can also look at the max frequency happened and have an array of that size, instead of size `n`. Recall that Bucket Sort is the sorting algorithm where items are placed at Array indexes based on their values (the indexes are called "buckets")
```py
from collections import Counter
class Solution:
    def frequencySort(self, s: str) -> str:
        if len(s) <= 1:
            return s
        
        freq = Counter(s)
        maxFreq = max(freq.values())

        bucketSort = [[] for _ in range(maxFreq + 1)]

        for char, fr in freq.items():
            bucketSort[fr].extend([char*fr])

        result = ''
        for i in range(len(bucketSort) - 1, -1, -1):
            result += ''.join(bucketSort[i])

        return result
```

### 169 Majority Element
- https://leetcode.com/problems/majority-element/description/
- To solve in Space `O(1)`, we can use **Boyer-Moore Voting Algorithm**:If we had some way of counting instances of the majority element as +1 and instances of any other element as −1, summing them would make it obvious that the majority element is indeed the majority element.
```py
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        count = 0
        majority = None

        for i in range(len(nums)):
            if count == 0:
                majority = nums[i]
                count += 1

            elif nums[i] == majority:
                count += 1
            else:
                count -= 1

        return majority
```

### 229 Majority Element II
- https://leetcode.com/problems/majority-element-ii/editorial/
**Intuition**
To figure out a O(1) space requirement, we would need to get this simple intuition first. For an array of length `n`:
- There can be at most one majority element which is more than `⌊n/2⌋` times.
- There can be at most two majority elements which are more than `⌊n/3⌋` times.
- There can be at most three majority elements which are more than `⌊n/4⌋` times. etc

```py
class Solution:
    def majorityElement(self, nums: List[int]) -> List[int]:
        if not nums:
            return []
        
        majority1 = None
        majority2 = None
        counter1, counter2 = 0, 0

        for num in nums:
            if majority1 == num:
                counter1 += 1

            elif majority2 == num:
                counter2 += 1

            elif counter1 == 0:
                majority1 = num
                counter1 += 1
            
            elif counter2 == 0:
                majority2 = num
                counter2 += 1

            else:
                counter1 -= 1
                counter2 -= 1

        # Now that we checked two majorities, we check if they occurred more than n//3
        count1, count2 = 0, 0
        for num in nums:
            if majority1 == num:
                count1 += 1
            if majority2  == num:
                count2 += 1

        result = []
        if count1 > len(nums) // 3:
            result.append(majority1)
        if count2 > len(nums) // 3:
            result.append(majority2)

        return result
```
