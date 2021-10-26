## 52. Find Median from a Data Stream
- You need to implement a data structure that will store a dynamically growing list of integers and provide efficient access to their median.
- **Solution**:
- We will assume that `x` is the median age of a user in a list. Half of the ages in the list will be smaller than (or equal to) `x`, and the other half will be greater than (or equal to) `x`. We can divide the list into two halves: one half to store the smaller numbers (say `smallList`), and one half to store the larger numbers (say `largeList`). The median of all ages will either be the largest number in the `smallList` or the smallest number in the `largeList`. If the total number of elements is even, we know that the median will be the average of these two numbers. The best data structure for finding the smallest or largest number among a list of numbers is a Heap.

Here is how we will implement this feature:

1. First, we will store the first half of the numbers (`smallList`) in a **Max Heap**. We use a **Max Heap** because we want to know the largest number in the first half of the list.

2. Then, we will store the second half of the numbers (`largeList`) in a **Min Heap**, because we want to know the smallest number in the second half of the list.

3. We can calculate the median of the current list of numbers using the top element of the two heaps.

- **Time Complexity**:
  - **Insert**: `O(logn)`
  - **Find Median**: `O(1)`
- **Space Complexity**: O(n)

```py
from heapq import *
class median_of_ages:

  maxHeap = []
  minHeap = []

  def insert_age(self, num):
    if not self.maxHeap or -self.maxHeap[0] >= num:
      heappush(self.maxHeap, -num)
    else:
      heappush(self.minHeap, num)

    if len(self.maxHeap) > len(self.minHeap) + 1:
      heappush(self.minHeap, -heappop(self.maxHeap))
    elif len(self.maxHeap) < len(self.minHeap):
      heappush(self.maxHeap, -heappop(self.minHeap))

  def find_median(self):
    if len(self.maxHeap) == len(self.minHeap):
      # we have even number of elements, take the average of middle two elements
      return -self.maxHeap[0] / 2.0 + self.minHeap[0] / 2.0

    # because max-heap will have one more element than the min-heap
    return -self.maxHeap[0] / 1.0


# Driver code

medianAge = median_of_ages()
medianAge.insert_age(22)
medianAge.insert_age(35)
print("The recommended content will be for ages under: " + str(medianAge.find_median()))
medianAge.insert_age(30)
print("The recommended content will be for ages under: " + str(medianAge.find_median()))
medianAge.insert_age(25)
print("The recommended content will be for ages under: " + str(medianAge.find_median()))
```
