
## 1. Merge k sorted Lists: https://leetcode.com/problems/merge-k-sorted-lists/

- always think a Brute-force approach first!
- Time complexity : `O(NlogN)` where N is the total number of nodes.
  - Collecting all the values costs O(N) time.
  - A stable sorting algorithm costs O(NlogN) time.
  - Iterating for creating the linked list costs O(N) time.
- Space complexity : O(N).

- `arr.sort()` sorts arr in-place. Time complexity is `O(nlog n)`. The function has two optional attributes which can be used to specify a customized sort:
  - `key: sorts the list based on a function or criteria`
  - `reverse: boolean, if true, sort in reverse order`

```py
arr.sort(key=abs, reverse=True)

# A callable function which returns the length of a string
def lengthKey(str):
  return len(str)

list2 = ["London", "Paris", "Copenhagen", "Melbourne"]
# Sort based on the length of the elements
list2.sort(key=lengthKey)
```

- The other approach is to compare every k nodes (head of every linked list) and get the node with the smallest value. Extend the final sorted linked list with the selected nodes. We can use PriorityQueue to save the first elements of all lists in a PriorityQueue and then retrieve the smallest value in the queue first and increment the relevant list node till we finish all of those lists.
- Time Complexity: `O(N log k)`, N: number of nodes in final list, k: number of linked lists. Finding a min value among k values in a priority queue is O(1). Inserting and poping which includes sorting in a priorityQueue for k values will be O(log k) and we have N total nodes. The comparison cost will be reduced to O(logk) for every pop and insertion to priority queue. But finding the node with the smallest value just costs O(1) time.

- `from Queue import PriorityQueue`, the `PriorityQueue` : The lowest valued entries are retrieved first. A typical pattern for entries is a tuple in the form: `(priority_number, data)`.
```py
from Queue import PriorityQueue

q = PriorityQueue()

q.put((3, 'Read')
q.put((5, "Write'))

OR 

q.put(4)
############

q.empty() # check if it's empty

############

val = q.get()
val, nn = q.get()
```
```py
from Queue import PriorityQueue

def mergeKLists(self, lists):
      """
      :type lists: List[ListNode]
      :rtype: ListNode
      """
      q = PriorityQueue()

      for l in lists:
          if l:
              q.put((l.val, l))

      dummy = head = ListNode(1000)       

      while not q.empty():
          val, node = q.get()
          head.next = ListNode(val)
          head = head.next

          node = node.next

          if node:
              q.put((node.val, node))

      return dummy.next

```

## 2. Design a data structure follows LRU cache (Least recently used cache).
https://leetcode.com/problems/lru-cache/
- The functions get and put must each run in O(1) average time complexity.
- We're asked to implement the structure which provides the following operations in O(1) time :
    - Get the key / Check if the key exists
    - Put the key
    - Delete the first added key

The first two operations in O(1) time are provided by the standard hashmap, and the last one - by linked list.

- The problem can be solved with a hashtable that keeps track of the keys and its values (linked list node) in the double linked list. One interesting property about double linked list is that the node can remove itself without other reference. In addition, it takes constant time to add and remove nodes from the head or tail.

One particularity about the double linked list that I implemented is that I create a pseudo head and tail to mark the boundary, so that we don't need to check the NULL node during the update. This makes the code more concise and clean, and also it is good for the performance. **We always insert or remove nodes between Head and Tail so we don't have to worry about checking for Null node**

```py
class DLNode(object):
    def __init__(self):
        self.key = -1
        self.val = -1
        self.next = None
        self.prev = None

class LRUCache(object):

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.capacity = capacity
        self.hashTable = {}
        self.size = 0
        
        self.head, self.tail = DLNode(), DLNode()
        
        self.head.next = self.tail
        self.tail.prev = self.head
        

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if key not in self.hashTable:
            return -1
        
        # key is in hashTable
        node = self.hashTable[key]
        
        # move that key,value node to start (after head) of Linked list
        self._removeNode(node)
        self._addNode(node)
        
        return node.val
        

    def put(self, key, val):
        """
        :type key: int
        :type value: int
        :rtype: None
        """
        if key not in self.hashTable:
            node = DLNode()
            node.key = key
            node.val = val
            self.hashTable[key] = node
            
            if self.size < self.capacity:
                self._addNode(node)
                self.size += 1
                
            else:
                # remove one node before tail
                # remove from dictionary
                keyRemove = self.tail.prev.key
                
                self._removeNode(self.tail.prev)
                self._addNode(node)
                
                del self.hashTable[keyRemove]
                
        else:
            # update value of that key in hashTable and Linked list
            curr = self.hashTable[key]
            curr.val = val
            self._removeNode(curr)
            self._addNode(curr)
            # print([(key, node.val) for key, node in self.hashTable.items()])
        
    def _addNode(self, node):
        node.next = self.head.next
        self.head.next = node
        
        node.next.prev = node
        node.prev = self.head
        
        
    def _removeNode(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev
        

# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)

```

## 3. Robot Bounded in Circle
- https://leetcode.com/problems/robot-bounded-in-circle/
- For example, `GGLLGG` after the first cycle, we're back to the origin but pointing different direction, after the second cycle, we're back to origin and pointing N. For the instruction which is a vector changes 2 things:
  - Change in position
  - Change in direction

- How do we know if the instructions causes robot to stuck in a cycle?
  1. if the change in the position is zero (after the entire iteration through the instructions we started at the origin and end at the origin)
  2. Both postion and direction change: suppose the position changes after the first iteration but if the direction also changes it means it'll be stuck in a loop. If the direction after the iteration does not change then there won't be a loop. Like `GR` after 4 iteration gets stuck in a loop (because of 4 directions).

- So the loop can appear after **0 or 2 (180 change in direction) or 4 (90 degree change in direction) iterations**. After at most 4 cycle if the postion change was zero we know that we stuck in a loop OR we can run iteration once and if:
  - if the position did not change after one iteration
  - if the postion and direction both change after one iteration.

```py
def isRobotBounded(self, instructions):
    """
    :type instructions: str
    :rtype: bool
    """

    direction = 0
    i, j = 0, 0


    for instruct in instructions:

        if instruct == 'L':
            direction = (direction - 1) % 4

        elif instruct == 'R':
            direction = (direction + 1 ) % 4

        elif instruct == 'G':

            if direction == 0: # North
                j += 1
            elif direction == 1: #East
                i += 1
            elif direction == 2: # South
                j -= 1
            else: # West
                i -= 1

    return (i,j) == (0,0) or (direction != 0)
    
 # Or base on direction
 def isRobotBounded(self, instructions):
    """
    :type instructions: str
    :rtype: bool
    """

    dirX, dirY = 0, 1
    x, y = 0, 0

    for instruct in instructions:

        if instruct == 'G':
            x, y = x + dirX, y + dirY

        elif instruct == 'L':
            dirX, dirY = -dirY, dirX
        else':
            dirX, dirY = dirY, -dirX
          

    return (x,y) == (0,0) or (dirX, dirY) != (0, 1)

            
```
## 4. Merge Intervals
- https://leetcode.com/problems/merge-intervals/

- Anonymous, `lambda` function in python:
```py
g = lambda x, y: x + y
# sort a list of arrays based on their first element:
arr.sort(key: lambda x: x[0])
```

```py
def merge(self, intervals):
    """
    :type intervals: List[List[int]]
    :rtype: List[List[int]]
    """
    # sort the intervals based on the starting value

    intervals.sort(key = lambda x: x[0])
    start = intervals[0][0]
    end = intervals[0][-1]
    result = []

    for arr in intervals:
        # print(start, end)
        # print(arr)
        # print('**********')
        if arr[0] > end:
            result.append([start, end])
            start = arr[0]
            end = arr[-1]

        elif arr[0] == end:
            end = arr[-1]

        elif arr[0] >= start:
            if arr[-1] > end:
                end = arr[-1]

    result.append([start, end])
    return result
```

## 5. Number of Islands
- https://leetcode.com/problems/number-of-islands/
- we should iterate through the 2D grid and for each point, first check if we've not visited it and if so, run either bfs or dfs to get all connected points and everytime, we get out of those methods, we increment a counter and at the end, we return the counter.
- bfs, go through the node based on the order they were added while dfs uses stack instead of queue and so it does not maintain the order. In this problem, it does not make any difference.

**BFS**
```py
import collections

def numIslands(grid):
    visited = set()
    rows = len(grid)
    cols = len(grid[0])
    count = 0
    
    for i in range(rows):
        for j in range(cols):
            node = (i,j)
            if node not in visited and (grid[i][j] == "1"):
                bfs(node, grid, visited)
                count += 1
    return count


def bfs(node, grid, visited):
    queue = collections.deque([node])
    
    while queue:
        curr = queue.popleft()
        if curr in visited: continue
        visited.add(curr)
        queue.extend(getNeighborNodes(curr, grid))
        
    return

def getNeighborNodes(node, grid):
    result = []
    i, j = node
    
    if i > 0 and grid[i-1][j] == "1": result.append((i-1,j))
    if j > 0 and grid[i][j-1] == "1": result.append((i,j-1))
    if i < len(grid) - 1 and grid[i+1][j] == "1": result.append((i+1,j))
    if j < len(grid[i]) - 1 and grid[i][j+1] == "1": result.append((i,j+1))
        
    return result
```

**DFS**
- Iterative: similar to the code above instead of queue use stack
- Recursive (think about the base case when the recursion returns):
```py

def numIslands(grid):
    visited = set()
    rows = len(grid)
    cols = len(grid[0])
    count = 0
    
    for i in range(rows):
        for j in range(cols):
            node = (i,j)
            if node not in visited and (grid[i][j] == "1"):
                dfs(node, grid, visited)
                count += 1
    return count


def dfs(node, grid, visited):
    i, j = node

    if i < 0 or j < 0 or i > len(grid) - 1 or j > len(grid[i]) - 1 or grid[i][j] != "1" or node in visited: return

    visited.add(node)
    dfs((i-1,j), grid, visited)
    dfs((i+1,j), grid, visited)
    dfs((i,j-1), grid, visited)
    dfs((i,j+1), grid, visited)

```

## 6. Trapping Rain water
https://leetcode.com/problems/trapping-rain-water/
- If find the absolute max in the array, then we know that all the heights on the left side of that max wall will trap water based on a localMax on their left side (coming from left to right). and all the height on the right side of the abs wall, will trap water based on the local max height coming from right to left.

```py
def trap(height):
    area = 0
    
    # find the global max wall so we can go from left and right up to that wall and calculate trapped water
    globalIdx, globalMax = findGlobalMax(height)
    
    # from left to right to global max height
    localMax = 0
    for i in range(0, globalIdx):
        if height[i] > localMax:
            localMax = height[i]
            
        area += localMax - height[i]
        
    # from right to left up to global max height
    localMax = 0
    for i in range(len(height)-1, globalIdx, -1):
        if height[i] > localMax:
            localMax = height[i]
            
        area += localMax - height[i]
        
    return area


def findGlobalMax(height):
    globalIdx, globalMax = 0, 0
    
    for i, h in enumerate(height):
        if h > globalMax:
            globalIdx, globalMax = i, h
    return (globalIdx, globalMax)


```
## 7. Longest Common Prefix 
- https://leetcode.com/problems/longest-common-prefix/
- **Approach 1: Horizontal scanning** LCP(S1, ..., Sn) = LCP(LCP(LCP(S1,S2), S3), ...Sn). Iterate through the list of strings and at each iteration i, find the longest common prefix of strings. If it is an empty one, end the algorithm. 
- Time Complexity: O(S), S: sum of all characters in all strings. In the worst case O(n*m), m: the avg lenght of a string.
- Space: O(1)

```py
def longestCommonPrefix(self, strs):
      """
      :type strs: List[str]
      :rtype: str
      """
      size = len(strs)
      if size == 0:
          return ""
      if size == 1:
          return strs[0]

      prefix = strs[0]

      for i in range(1, size):
          word = strs[i]

          while (word.find(prefix) != 0):
              prefix = prefix[0:-1]


          if prefix == "": return ""
      return prefix
```

- The issue with horizantal scanning is that if a very short string (which is also a LCP) is at the end of the list, the above approach still does S comparisons. One way to optimize is to scan vertically. 
- Another approach is Trie:
  - Create a TrieNode and Trie and insert all words in it. Creating a prefix-based tree.
  - Create a method LCP in the Trie so that it return the longest common prefix. For that we walk on the trie and go deeper until we find a node having more than **1 children (branching)** and **end of the word**.
  - Time complexity: for n string in a list and m largest lenght of the word: O(n * m)
  - Space: O(n * m)

```py
class TrieNode(object):
    def __init__(self):
        self.children = {}
        self.endOfWord = False
        
class Trie(object):
    def __init__(self):
        self.root = TrieNode()
        
    def insert(self, word):
        curr = self.root
        
        for char in word:
            if char in curr.children:
                curr = curr.children[char]
                
            else:
                curr.children[char] = TrieNode()
                curr = curr.children[char]
        curr.endOfWord = True
        
        return True
    
    def _childrenCount(self, node):
        return len(node.children.keys())
    
    def lcp(self):
        curr = self.root
        prefix = ""
        
        while (self._childrenCount(curr) == 1 and curr.endOfWord == False):
            key = curr.children.keys()[0]
            prefix += key
            curr = curr.children[key]
        
        return prefix
        
def longestCommonPrefix(self, strs):
      """
      :type strs: List[str]
      :rtype: str
      """
      # Construct Trie
      trieTree = Trie()

      for word in strs:
          trieTree.insert(word)

      return trieTree.lcp()

```
