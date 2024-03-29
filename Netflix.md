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
- Time complexity: `O(logn)`
  - Inserting number to heap: `O(log n)`
  - Finding median from heap: `O(1)`
- Space complexity: `O(n)`


## Feature # 4: For efficiently distributing content to different geographic regions and for program recommendation to viewers, we want to determine titles that are gaining or losing popularity scores.
- We’ll be provided with a list of integers representing the popularity scores of a movie collected over a number of weeks. We need to identify only those titles that are either increasing or decreasing in popularity, so we can separate them from the fluctuating ones for better analysis.

- https://leetcode.com/problems/monotonic-array/

```py
def isMonotonic(self, nums: List[int]) -> bool:
        increase = decrease = True
        
        for i in range(len(nums) - 1):
            if nums[i] > nums[i + 1]:
                increase = False
            if nums[i] < nums[i + 1]:
                decrease = False
                
        return increase or decrease
```

## Feature # 5: For the client application, we want to implement a cache with a replacement strategy that replaces the least recently watched title.
- You need to come up with a data structure for this feature. Let’s break it down. If we think it through, we realize the following: i) This data structure should maintain titles in order of time since last access; ii) If the data structure is at its capacity, an insertion should replace the least recently accessed item.

**Solution:** 
(https://leetcode.com/problems/lru-cache/discuss/45911/Java-Hashtable-%2B-Double-linked-list-(with-a-touch-of-pseudo-nodes))
- The problem can be solved with a hashtable that keeps track of the keys and its values in the double linked list. One interesting property about double linked list is that the node can remove itself without other reference. In addition, it takes constant time to add and remove nodes from the head or tail.

- One particularity about the double linked list that I implemented is that I create a pseudo head and tail to mark the boundary, so that we don't need to check the NULL node during the update. This makes the code more concise and clean, and also it is good for the performance.
- https://leetcode.com/problems/lru-cache/

```py
class DbNode:
    def __init__(self, key=0, val=0):
        self.key = key
        self.val = val
        self.next = None
        self.prev = None


class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._size = 0
        self._hashTable = {}
        #DB linked list
        self._head = self._tail = DbNode(0)
        self._head.next = self._tail
        self._tail.prev = self._head
        

    def get(self, key: int) -> int:
        if key in self._hashTable:
            node = self._hashTable[key]
            self._removeNode(node)
            self._moveNodeStart(node)
            
            return node.val
        return -1
        

    def put(self, key: int, val: int) -> None:
        if key in self._hashTable:
            node = self._hashTable[key]
            self._removeNode(node)
            self._moveNodeStart(node)
            
            # update value
            node.val = val
            
        else:
            
            if self._size >= self.capacity:
                LRUNode = self._tail.prev
                self._removeNode(LRUNode)
                self._size -= 1 
                del self._hashTable[LRUNode.key]
               
            # add new node
            newNode = DbNode(key, val)
            self._moveNodeStart(newNode)
            self._hashTable[key] = newNode
            self._size += 1
        
    
    def _moveNodeStart(self, node):
        node.next = self._head.next
        self._head.next = node
        node.next.prev = node
        node.prev = self._head
        
    def _removeNode(self, node):
        node.next.prev = node.prev
        node.prev.next = node.next

        


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```

- Time complexity: `O(1)`
  - Get hashMap: O(1)
  - Set hashMap: O(1)
  - Insertion and deletion of Doubly-linked list: O(1)
  
- Space complexity: `O(n)`, n is the size of cache.


## Feature #6: Fetch Most Frequently Watched Titles
- https://leetcode.com/problems/lfu-cache/
- We need:
  1. hash table with key=key and value=linked list node to retrieve node O(1)
  2. hash table with key=freq and value= a linked list of nodes with that frequency
  3. min_freq to keep the min_freq value across all nodes

- when we add a new node: add it to the head and set min_freq=1
- when we visit an exisitng node or update an existing node: remove node from old linked list frequncy and add it to the new one with new freq + 1
- if the old linked list is now zero size and the old freq was min_freq, we increment min_freq by 1.

```py
import collections

class DBLNode:
    def __init__(self, key, val):
        self.val = val
        self.key = key
        self.freq = 1
        self.next = None
        self.prev = None
        
class DLinkedList:
    """
    An implementation of Doubly-Linked list
    
    two APIs provided:
    
    1. append(node): append the node to the head of the linked list
    
    2. pop(node = None): remove the node. If None is provided, remove
                        one from the tail which is the least recently used
    """
    def __init__(self):
        self.head = self.tail = DBLNode(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head
        self._size = 0
        
    def __len__(self):
        return self._size
    
    def append(self, node):
        node.next = self.head.next
        self.head.next = node
        node.next.prev = node
        node.prev = self.head
        
        self._size += 1
        
    def pop(self, node=None):
        if self._size == 0:
            return 
        
        if not node:
            node = self.tail.prev
        
        node.prev.next = node.next
        node.next.prev = node.prev
        
        self._size -= 1
        
        return node

class LFUCache:

    def __init__(self, capacity: int):
        """
        3 Things to maintain:
        
        1. a dict (self._hashNode) for the reference of all nodes given key
        That is O(1) time to retrieve node given a key.
        
        2. a dict (self._hashFreq) where key is the frequency and value if a doubly linked list
        
        3. The min frequency through all nodes, we can maintain it O(1) time, 
        taking advantage of the fact that the frequency can only increment by 1, Use the 
        following two rules:
        
            Rule 1: Whenever we see the size of the DLinkedList of current min frequency is 0,
            the min frequency must increment by 1
            
            Rule 2: Whenevr put in a new (key, value), the min frequency must be 1 (the new node)
        """
        
        self.capacity = capacity
        self._size = 0
        
        self._hashNode = dict()
        self._hashFreq = collections.defaultdict(DLinkedList)
        self._minFreq = 0
        
        
    def get(self, key: int) -> int:
        if key not in self._hashNode:
            return -1
        
        node = self._hashNode[key]
        self._update(node)
        
        return node.val
        

    def put(self, key: int, value: int) -> None:
        
        if self.capacity == 0:
            return
        
        if key in self._hashNode:
            node = self._hashNode[key]
       
            self._update(node)
            node.val = value
        
        else:
            if self._size == self.capacity:
                node = self._hashFreq[self._minFreq].pop()

                del self._hashNode[node.key]
                self._size -= 1
                
            
            node = DBLNode(key, value)
            self._hashNode[key] = node
            self._hashFreq[1].append(node)
            self._minFreq = 1
            self._size += 1

    
    def _update(self, node):
        """
        for the existing node, we need to update the frequency of the node in the DLinked list
        1. pop the node from old DLinkedList (with freq f)
        2. append the node to new DLinkedList (with freq f+1)
        3. if old DLinked list has size 0 and self._minFreq is f, update self._minFreq to f+1
        """
        freq = node.freq
        
        
        self._hashFreq[freq].pop(node)
        
        if self._minFreq == freq and not self._hashFreq[freq]:
            self._minFreq += 1
        
        node.freq += 1
        self._hashFreq[node.freq].append(node)


# Your LFUCache object will be instantiated and called as such:
# obj = LFUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```

- Time complexity: O(1) for all get, put, append and pop in linked list
- Space complexity: `O(n)` n: capacity of cache

## Feature # 7: During a user session, a user often “shops” around for a program to watch. During this session, we want to let them move back and forth in the history of programs they’ve just browsed. As a developer, you can smell a stack, right? But, we also want the user to be able to directly jump to the top-ranked program from the one’s they’ve browsed.
- In this feature, the user will be able to randomly browse through movie titles and read their summaries and reviews. We want to enable a Back button so the user can return to the previous title in the viewing history. We also want the user to immediately get the title with the highest viewer rating from their viewing history. We want to implement both of these operations in constant time to provide a good user experience.
- We’ll be provided with a sequential input of ratings to simulate the user viewing them one by one. For simplicity, we’ll assume that the movie ratings are all unique.
- https://leetcode.com/problems/min-stack/

- To enable a Back button, we need to store the user’s viewing history in some data structure. Pressing the Back button fetches the last viewed item. This indicates a last in first out (LIFO) structure, which is characteristic of a stack. In a stack, push and pop operations can be easily implemented in O(1)
O(1)
. However, the stack doesn’t allow random access to its elements, let alone access to the element with the maximum rating. We will need to create a stack-like data structure that offers a getMax operation, in addition to push and pop, that all run in O(1)
O(1)
.

The implementation of such a data structure can be realized with the help of two stacks: `max_stack` and `main_stack`. The `main_stack` holds the actual stack with all the elements, and `max_stack` is a stack whose top always contains the current maximum value in the stack.

How does it do this? The answer is in the implementation of the push function. Whenever push is called, `main_stack` simply inserts it at the top. However, `max_stack` checks the value being pushed. If `max_stack` is empty, this value is pushed into it and becomes the current maximum. If `max_stack` already has elements in it, the value is compared with the top element in this stack. The element is pushed into `max_stack` if it is greater than the top element else, the top element is pushed again.

The `pop()` function simply pops off the top element from `main_stack` and `max_stack`.

Due to all these safeguards, the `max_rating` function only needs to return the value at the top of `max_stack`.

```py
class Stack:
    def __init__(self):
        self._stack = []
        
    def push(self, val):
        self._stack.append(val)
        
        
    def pop(self):
        if self.isEmpty():
            return None
        
        return self._stack.pop()
        
        
    def top(self):
        if self.isEmpty():
            return None
        
        return self._stack[-1]
        
    def isEmpty(self):
        return len(self._stack) == 0
    
    def size(self):
        return len(self._stack)

class MinStack:

    def __init__(self):
        self._mainStack = Stack()
        self._minStack = Stack()
        

    def push(self, val: int) -> None:
        self._mainStack.push(val)
        
        if self._minStack.isEmpty() or self._minStack.top() > val:
            self._minStack.push(val)
        else:
            self._minStack.push(self._minStack.top())
        

    def pop(self) -> None:
        self._mainStack.pop()
        self._minStack.pop()
        

    def top(self) -> int:
        return self._mainStack.top()
        

    def getMin(self) -> int:
        if self._minStack.isEmpty():
            return None
        
        return self._minStack.top()
```

## Feature # 8: As you beta tested feature #7, a user complained that the next and previous functionality isn’t working correctly. Using their session history, we want to check if our implementation is correct or indeed buggy.
- We’ll receive two lists of push and pop operations. These lists will contain the ID’s of the pages that were browsed. We want to verify whether our implementation of the max stack is behaving correctly or not. To do this, we can check if the sequence of push operations and the sequence of pop operations have been interleaved and performed on a valid stack that was initially empty.
- https://leetcode.com/problems/validate-stack-sequences/

We only have the push and pop operations and not the timestamps for when they were performed. Since the user did not browse any title more than once and the Back button is disabled at the end. This means that every ID that was pushed to the stack must have been popped once.

After every push operation, we immediately try to pop. If the session is fine, all pushed elements will get popped at one point.

Here is how the implementation will take place:

1. Declare an empty stack.

2. Remove the element from the front of the pushed list and push it onto the stack.

3. If the element at the top of the stack is the same as the item at the front of the popped list, pop the element from the stack and remove it from the popped list.

4. If the stack is empty by the end, return True.

Otherwise, return False.

```py
class Solution:
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
        
        if len(pushed) != len(popped):
            return False
        
        stack = []
        j = 0
        
        for ele in pushed:
            stack.append(ele)
            
            while stack and popped[j] == stack[-1]:
                stack.pop()
                j += 1
            
            
        return len(stack) == 0
```
- Time complexity: `O(n)` n: the size of the pushed or popped stack. because n elements will be pushed, and n elements will get popped.
- Space complexity: `O(n)`. In the worst case, all n elements will be pushed into the stack, so the space complexity will be O(n).


## Feature # 9: Created all the possible viewing orders of movies appearing in a specific sequence of genre.
- Suppose we have n different movie genres like Action, Comedy, Family, Horror, and so on. In each genre, we have 0 to k movies. We need to screen several movies of different genres in a specific sequence. For example, in the morning, may want to feature a Family program, followed by a Comedy. Late at night, we may want to show Horror movies followed by Action movies. At prime time, we may want to use some other sequence.
- If we have the movies given in the example above and the input to our algorithm is ["Family", "Action"], then it should return the following list containing all of the possible combinations of a Family movie followed by an Action movie, chosen from the available data. We add a semicolon (;) just to separate the movie names.
```py
["Frozen;Iron Man;","Frozen;Wonder Woman;","Frozen;Avengers;","Kung fu Panda;Iron Man;","Kung fu Panda;
Wonder Woman;","Kung fu Panda;Avengers;","Ice Age;Iron Man;","Ice Age;Wonder Woman;","Ice Age;Avengers;"]
```
- https://leetcode.com/problems/letter-combinations-of-a-phone-number/


To solve this problem, we can use a backtracking algorithm template to generate all of the possible combinations correctly.

Let’s break down this problem into four parts.

- If the input has only one genre, return all the movies in that genre—for example, `["Action"]`. This example is trivial where all of the corresponding movies will be returned, for example, `["Iron Man", "Wonder Woman", "Avengers"]`.

- We already know how to solve the one-genre problem. To solve the two-genre problem, such as `["Action, Family"]`, we can pick the one-genre solutions for the Action genre and then append the solutions for the one-genre problem of the Family genre. For example, start with `"Iron Man"` and then append all the solutions for the one-genre problem of the Family genre, resulting in `[[Iron Man, Frozen],[Iron Man, Kung Fu Panda],[Iron Man, Ice Age]]`. Then, we can switch to the other solutions for the one-genre problem of the `Action` genre and repeat.

After making the above-stated observations, our algorithm will look like this:

- We return an empty array if the input is an empty string.

- We initialize a data structure (for example, a dictionary) that maps the genres to their movies. For example, we map `"Action"` to `"Iron Man"`, `"Wonder Woman"`, and `"Avengers"`.

- We initialize a backtracking function and utilize it to generate all possible combinations.

- Two primary parameters will be passed to the function: the `path` of the current combination of movies and the `index` of the given genre array.

- If our current combination of movies is the same length as the input, then we have an answer. Therefore, add it to our answer and backtrack.

- Otherwise, we get all the movies that correspond with the current genre we are looking at: `genre[index]`.

- We then loop through these movies. We add each movie to our current path and call backtrack again, but move on to the next genre by incrementing the index by 1. We make sure to remove the movie from the path once we are finished with it.

```py
def movie_combinations(categories):
    if len(categories) == 0:
        return []
    
    movies = {
        "Family": ["Frozen", "Kung fu Panda", "Ice Age"],
        "Action": ["Iron Man", "Wonder Woman", "Avengers"],
        "Fantasy": ["Jumangi", "Lion King", "Tarzan"],
        "Comedy": ["Coco", "The Croods", "Vivi", "Pets"],
        "Horror": ["Oculus", "Sinister", "Insidious", "Annebelle"],
    }
    
    def backtrack(index, path):
        """
        path: ['Frozen;', 'Iron Man;']
        categories: ["Action", "Family"]
        """
        if len(path) == len(categories):
            print(path)
            combinations.append("".join(path))
            return
        
        possibleMovies = movies[categories[index]]
        
        if possibleMovies:
            for movie in possibleMovies:
                path.append(movie + ";")
                backtrack(index + 1, path)
                path.pop()
            
    combinations = []
    backtrack(0, [])
    return combinations
    
# Example 2
categories = ["Family", "Action"]
combinations = []
print("Output 2:")
print(movie_combinations(categories))
```
- Time Complexity: `O(k^n * n)` Here `n` is the number of genres and `k` refers to the maximum-value length in the dictionary, not the length of the input. The worst-case is when the input only consists of the `Comedy` and `Horror genres`. In this case, we have to visit `k` additional paths for each extra genre. To create a combination, each combination will cost up to `n`. We can generalize this problem to a scenario where genres correspond with up to m movies. In this case, the time complexity will be O(m^n * n).
- Space Complexity: `O(k*n)`: where n is the total number of genres and k is the number of movies in each genre. The algorithm takes n space, because of the recursive call. In the worst case, we will be n levels deep in the call stack. The dictionary takes O(k) space because there are at most k movies of n
different genres.

## Feature #10: Calculate Median of Buffering Events
- https://leetcode.com/problems/sliding-window-median/

- To solve this problem we use the same idea as used in **## Feature 3. Find Median Age**. The only additional requirement is removing the outgoing elements from the window. Assume that we are using two heaps as mentioned in the Find Median Age problem, but only the tops of heaps are accessible. Deleting elements that are not on the top of the heap is an **O(log n)** operation. We need to find an efficient way to remove elements that are moving out of the window.

If the two heaps are balanced, only the top is needed to find the medians. Keeping the heaps balanced will allow us to keep invalidated elements in the heap without interfering with the results.

**Note: Here, balancing the heaps doesn’t mean balancing the actual sizes of heaps. Balancing heaps refers to the count of valid elements, as we are only concerned with valid elements.**

To do this, we can use the lazy removal technique, i.e., utilizing a dictionary to keep track of invalidated elements, and once they reach heap tops, remove them from the heaps.

The most challenging part here is to keep the heap balanced while keeping the invalidated elements. We can move the invalidated elements from one heap to the heap that contains invalidated elements.

**Algorithm**
For this problem, we need to find the median in a sliding window of size `k`. First, we will store the first half of the numbers in a Max Heap (`small_list`). We use a Max Heap because we want to know the largest number in the first half of the list. Then, we will transfer the top `k / 2` elements from Max Heap (`small_list`) to the Min Heap (`large_list`) because we want to know the smallest number in the second half of the list.

In the case of an odd-sized window, the top of the Max Heap contains the median. Otherwise, the median is the arithmetic mean of the tops of the two heaps. As the window slides, one new number will enter the window and one number will leave the window. If the incoming number is smaller than the current top of the Max Heap, it must be inserted in the Max Heap. This is because it belongs to the smaller half (by sorted order) of the numbers in the sliding window. Otherwise, it needs to be inserted in the Min Heap. We use the variable named `balance` to check if ceil(`k/2`) members belonging to the sliding window are present in the Max Heap. If not, we transfer the top element from the Max Heap to the Min Heap. The second highest element then springs to the top of the max heap.

Similarly, if more than floor(`k/2`) elements end up in the Min Heap, then we restore the balance by popping the smallest element from the Min Heap and adding it to the Max Heap. This way, the top of the Max Heap always has the median (in case of odd valued `k`). We don’t immediately remove outgoing elements from either heap. Instead, we use a hash table (`hash_map`) to keep tabs on values that are no longer in the sliding window. If such values don’t end up on top of the heaps, they don’t interfere with correct median calculation. If such values do end up on top of the heap, we lazily remove them.

```py
from heapq import heappop, heappush, heapify

def median_bufering_events(nums, k):
    # Will store the medians
    medians = []

    # Hash-map will keep track of invalid numbers
    hash_map = {}

    #max heap
    small_list = []
    heapify(small_list)

    #min heap
    large_list = []
    heapify(large_list)

    # Index of current incoming element being processed
    i = 0

    # Initialize the small_list heap
    # Muliplying each elemnt by -1 to implement max heap.
    for i in range(0,k):
        heappush(small_list, -1 * nums[i])

    i+=1 
    # Initialize the large_list heap 
    for j in range(0,k//2):
        element = heappop(small_list)
        heappush(large_list, -1 * element)

    # Start an infinite loop
    while True:
        if (k&1) == 1:
            medians.append(float(small_list[0] * -1))
            
        else:
            temp = (float(small_list[0]* -1) + float(large_list[0])) * 0.5
            medians.append(temp)
            
        # Break the loop if all of the elements are processed
        if i >= len(nums):
            break
      
        # Outgoing element
        out_num = nums[i - k]

        # Incoming element
        in_num = nums[i]
        i+=1
        
        # Balance factor
        balance = 0

        # Number `out_num` exits window
        if out_num <= (small_list[0] * -1):
          balance -= 1
        else:
          balance += 1

        # If the outgoing element is not present in the hash-map 
        # store the `out_num` in the hash-map with value 1,
        # otherwise increment the count of `out_num` in the hash-map.

        if out_num in hash_map:
          hash_map[out_num] = hash_map[out_num] + 1
        else:
          hash_map[out_num] = 1

        # number `in_num` enters window
        if small_list and in_num <= (small_list[0] * -1):
          balance+=1
          heappush(small_list, in_num * -1)
        else:
          balance-=1
          heappush(large_list, in_num)
  
        # Re-balance small_list
        if balance < 0:
          heappush(small_list, (-1 * large_list[0]))
          heappop(large_list)
          balance +=1

        # Re-balance large_list
        if balance > 0:
          heappush(large_list, (-1 * small_list[0]))
          heappop(small_list)
          balance +=1

        # Remove invalid numbers that should be discarded from small_list heap tops
        while (small_list[0] * -1) in hash_map and (hash_map[(small_list[0] * -1)] > 0):
            hash_map[small_list[0] * -1] = hash_map[small_list[0] * -1 ] - 1
            heappop(small_list)

        # Remove invalid numbers that should be discarded from large_list heap tops
        while large_list and large_list[0]  in hash_map and (hash_map[large_list[0]] > 0):
            hash_map[large_list[0]] = hash_map[large_list[0]] - 1
            heappop(large_list)
      
    # Return medians
    return medians

print("Example - 1")
arr = [1,3,-1,-3,5,3,6,7]
k = 3
print("Input: array =", arr,  ", k = " , k)
output = median_bufering_events(arr, k)
print("Output: Medians =" ,output)

print("Example - 2")
arr2 = [1,2]
k = 1
print("Input: array =" ,arr2 , ", k = " , k)
output2 = median_bufering_events(arr2, k)
print("Output: Medians =", output2)
```

- **Time complexity: `O(n logk)` **
    - Either or sometimes both of the heaps get every elements inserted into it at least once. Hence, the `O(log k)` insertions are repeated for each of the `n` values.
    - From the top of the heaps, `O(n - k)` removals (the number of sliding windows instances) take place. Each of those takes `O(log k)`
    - Hash table operations are take `O(1)`
    
- **Space complexity: `O(n)`**
  - it takes `O(k) + O(n) ~ O(n)` linear space
  - the heaps collectively require `O(k)` space. The hash table needs `O(n - k)` space.

## Feature # 11: A set of movies needs to be presented in different orders to a different set of users. We want to generate all the possible permutations of movies in a given marathon.
- https://leetcode.com/problems/permutations/

Backtracking is an algorithm for finding all solutions by exploring all potential candidates. If the solution candidate turns to be not a solution (or at least not the last one), backtracking algorithm discards it by making some changes on the previous step, i.e. backtracks and then try again.

**Solution 1**
- Take a look at my `BackTracking.md` for the pattern of backtracking.
```py
def permutation():
    if len(nums) == 0:
        return []

    if len(nums) == 1:
        return [nums]

    def helper(node, seen, curr):
        if len(curr) == len(nums):
            result.add(tuple(curr))
            return

        for num in nums:
            if num in seen:
                continue

            seen.add(num)
            curr.append(num)

            helper(num, seen, curr)

            seen.remove(num)
            curr.pop()

        return


    result = set()

    for num in nums:
        helper(num, set(), [])

    return list(result)
```



Here is a backtrack function which takes the index of the first integer to consider as an argument `backtrack(first)`.

- If the first integer to consider has index n that means that the current permutation is done.
- Iterate over the integers from index `first` to index `n - 1`.
  - Place `i-t`h integer first in the permutation, i.e. `swap(nums[first], nums[i])`.
  - Proceed to create all permutations which starts from `i-th` integer : `backtrack(first + 1)`.
  - Now backtrack, i.e. `swap(nums[first], nums[i])` back.

```py
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        
        def backtrack(first):
            if first == n:
                output.append(nums[:])
                
            for i in range(first, n):
                # place ith integer first
                # in the current permutation
                nums[first], nums[i] = nums[i], nums[first]
                
                #use next integers to complete the permutations
                backtrack(first + 1)
                
                #backtrack
                nums[first], nums[i] = nums[i], nums[first]
            
        n = len(nums)
        output = []
        backtrack(0)
        
        return output
```
- Time complexity: `O(n!)` OR `k-permutation_of_n, or partial permutation`: `P(N,k) = N!/(N-k)! = N(N-1)...(N-k+1)` and `N! <= P(N, k) <= N * N!`
  - For the complexity, I think you can explain in this way: in the first level of the tree, you have N options and for each of the option, you have N-1 option, and for each of these N-1 options, you have another N-2 options, so putting them together you would end up N*(N-1)*(N-2).... = N!
- Space complexity: `O(n)` the maximum stack depth is `n`, the height of any branch from the root to any leaf.

## Feature #12: Maintain Continue Watching Bar
- https://leetcode.com/problems/maximum-frequency-stack/

solution:
As we care about the frequency of the show, let’s save the number of occurrences for each show in a map named `frequency`.

To pop the element with the maximum frequency we must keep track of the current maximum frequency element in the stack. We can use a variable named `max_frequency` to perform this task.

Let’s save the elements for each frequency in a map named `group`, in which we can save multiple elements for each frequency. The map, for each frequency, will store an array that will work as a stack to push or pop an element.

In the case, where multiple elements have the same frequencies, we can use `group` to pop the most recent element — the top of the stack.

```py

from collections import Counter
from collections import defaultdict

class FreqStack(object):

    def __init__(self):
        self.freq = Counter()
        self.group = defaultdict(list)
        self.maxfreq = 0

    def push(self, x):
        f = self.freq[x] + 1
        self.freq[x] = f
        if f > self.maxfreq:
            self.maxfreq = f
        self.group[f].append(x)

    def pop(self):
        x = self.group[self.maxfreq].pop()
        self.freq[x] -= 1
        if not self.group[self.maxfreq]:
            self.maxfreq -= 1

        return x
```
- Time complexity: `O(1)` for both push and pop
- Space complexity: `O(n)`
