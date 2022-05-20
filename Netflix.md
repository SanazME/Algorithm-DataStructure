## Feature # 1: We want to enable users to see relevant search results despite minor typos.
- it means that we're given a list of titles (words) and we need to group the titles that are anagrams of eachother (An anagram is a word, phrase, or sentence formed from another word by rearranging its letters.)
- we create a hashMap with 
  - key: a tuple of 26 indices (related to 26 letters in alphabet) and each value in a index corresponds to frequency that letter is used in a word: `tuple(0,1,..,3)`
  - value: list of words that are anagrams (have the same frequency of letters and the same letters)

- `{(0,0..,1..): ['duel', 'dule']}`

```py
def searchSimilarTitles(titles):
    hashMap = {}
    
    for title in titles:
        freq = [0 for _ in range(26)]
        
        for char in title:
            index = ord(char) - ord('a')
            freq[index] += 1
        
        key = tuple(freq)
        
        if key in hashMap:
            hashMap[key].append(title)
        else:
            hashMap[key] = [title]
        
    print(hashMap)
    return list(hashMap.values())

titles = ["duel","dule","speed","spede","deul","cars"]
print(searchSimilarTitles(titles))
```
- `Time complexity: O(n*k)`: Let n be the size of the list of strings, and k be the maximum length that a single string can have.
- `Space complexity: O(n*k)`: we still save all the strings in dictionary

## Feature # 2: Enable the user to view the top-rated movies worldwide, given that we have movie rankings available separately for different geographic regions.
- We’ll be given n lists that are all sorted in ascending order of popularity rank. We have to combine these lists into a single list that will be sorted by rank in ascending order, meaning from best to worst.

**1. Brute Force**
  - Traverse all the linked lists and collect the values of the nodes into an array.
  - Sort and iterate over this array to get the proper value of nodes.
  - Create a new sorted linked list and extend it with the new nodes.
```py
def mergeKLists(lists):
    """
    :type lists: List[ListNode]
    :rtype: ListNode
    """
    nodes = []
    
    for l in lists:
        while l:
            nodes.append(l.val)
            l = l.next
            
    head = point = ListNode(0)
    for x in nodes.sort():
        point.next = ListNode(x)
        point = point.next
    return head.next
```
- `Time complexity : O(NlogN)` where N is the total number of nodes.
  - Collecting all the values costs `O(N)` time.
  - A stable sorting algorithm costs `O(NlogN)` time. (merge sort in `nodes.sort()`)
  - Iterating for creating the linked list costs `O(N)` time.

- `Space complexity : O(N)`
  - Sorting cost `O(N)` space (depends on the algorithm you choose).
  - Creating a new linked list costs `O(N)` space.

**2. PriorityQueue**
- The other approach is to compare every k nodes (head of every linked list) and get the node with the smallest value. Extend the final sorted linked list with the selected nodes. We can use PriorityQueue to save the first elements of all lists in a PriorityQueue and then retrieve the smallest value in the queue first and increment the relevant list node till we finish all of those lists.

```py
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

from queue import PriorityQueue
class Solution:
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        if not lists:
            return None
    
        n = len(lists)

        if n == 1:
            return lists[0]

        curr = head = ListNode(0)
        q = PriorityQueue()
        i = 0

        for node in lists:
            if node:
                i += 1
                q.put((node.val, i, node))

        while not q.empty():
            i += 1
            node = q.get()[2]
            curr.next = node

            node = node.next
            curr = curr.next

            if node:
                q.put((node.val, i, node))

        return head.next
```


- `Time complexity : O(Nlogk)` where `k` is the number of linked lists.
  - The comparison cost will be reduced to `O(logk)` for every pop and insertion to priority queue. But finding the node with the smallest value just costs `O(1)` time.
  - There are `N` nodes in the final linked list.

- Space complexity : 
  - `O(n)` Creating a new linked list costs `O(n)` space.
  - `O(k)` The code above present applies in-place method which cost `O(1)` space. And the priority queue **(often implemented with heaps)** costs `O(k)` space (it's far less than `N` in most situations).


## Feature # 3: As part of a demographic study, we are interested in the median age of our viewers. We want to implement a functionality whereby the median age can be updated efficiently whenever a new user signs up for Netflix.
- We will have a stream of data and we need to output median in real-time as data is added.
- https://leetcode.com/problems/find-median-from-data-stream/

**Solution:** We will assume that `x` is the median age of a user in a list. Half of the ages in the list will be smaller than (or equal to) `x`, and the other half will be greater than (or equal to) `x`. We can divide the list into two halves: one half to store the smaller numbers (say `smallList`), and one half to store the larger numbers (say `largeList`). The median of all ages will either be the largest number in the `smallList` or the smallest number in the `largeList`. If the total number of elements is even, we know that the median will be the average of these two numbers. The best data structure for finding the smallest or largest number among a list of numbers is a **Heap**: https://www.educative.io/edpresso/what-is-a-heap

Here is how we will implement this feature:

1. First, we will store the first half of the numbers (`smallList`) in a Max Heap. We use a Max Heap because we want to know the largest number in the first half of the list.

2. Then, we will store the second half of the numbers (`largeList`) in a Min Heap, because we want to know the smallest number in the second half of the list.

3. We can calculate the median of the current list of numbers using the top element of the two heaps.

```py
from heapq import *

class MedianFinder:

    def __init__(self):
        self.maxHeap = []
        self.minHeap = []
        

    def addNum(self, num: int) -> None:
        if not self.maxHeap or -self.maxHeap[0] >= num:
            heappush(self.maxHeap, -num)
        else:
            heappush(self.minHeap, num)
            
        # making sure the lenght of two arrays are either equal or maxHeap has only one element more
        if len(self.maxHeap) > len(self.minHeap) + 1:
            heappush(self.minHeap, -heappop(self.maxHeap))
        elif len(self.maxHeap) < len(self.minHeap):
            heappush(self.maxHeap, -heappop(self.minHeap))
        

    def findMedian(self) -> float:
        if len(self.maxHeap) == len(self.minHeap):
            return (-self.maxHeap[0] + self.minHeap[0]) / 2.0
        
        return - self.maxHeap[0] / 1.0


# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()
```

