
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

   # direction: N, E, S, W
    directions = [(1,0), (0,1),(-1,0),(0,-1)] # di, dj

    currDir = 0
    i, j = 0, 0
    for instruction in instructions:
        if instruction == "L":
            currDir = (currDir - 1) % 4
        elif instruction == "R":
            currDir = (currDir + 1) % 4
        else:
            di, dj = directions[currDir]
            i += di
            j += dj
    return (i,j)==(0,0) or currDir != 0
    
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

## 4.1 Meeting Rooms II
- https://leetcode.com/problems/meeting-rooms-ii/
- we use min heap or priority Queue to save the end time of meetings 
- Algorithm
	1. Sort the given meetings by their start time.
	2. Initialize a new min-heap and add the first meeting's ending time to the heap. We simply need to keep track of the ending times as that tells us when a meeting room will get free.
	3. For every meeting room check if the minimum element of the heap i.e. the room at the top of the heap is free or not.
	4. If the room is free, then we extract the topmost element and add it back with the ending time of the current meeting we are processing.
	5. If not, then we allocate a new room and add it to the heap.
	6. After processing all the meetings, the size of the heap will tell us the number of rooms allocated. This will be the minimum number of rooms needed to accommodate all the meetings.
- Time complexity: `O(N logN)`:
	- There are two major portions that take up time here. One is sorting of the array that takes `O(NlogN)` considering that the array consists of N elements.
	- Then we have the min-heap. In the worst case, all N meetings will collide with each other. In any case we have N add operations on the heap. In the worst case we will have N extract-min operations as well. Overall complexity being `(NlogN)` since extract-min operation on a heap takes `O(logN)`.

- Space Complexity: `O(N)` because we construct the min-heap and that can contain N elements in the worst case as described above in the time complexity section. Hence, the space complexity is `O(N)`.

```py
import heapq
class Solution(object):
    def minMeetingRooms(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: int
        """
        
        if len(intervals) <= 0:
            return 0
        
        heap = []
        # sort the meetings in increasing order of their start time
        intervals.sort(key = lambda x: x[0])
        
        # Add the firt meesting end time to the heap
        heappush(heap, intervals[0][1])
        
        for i in range(1, len(intervals)):
            
            # if the room due to free up the earliest is free, assign that room to his meeting
            if heap[0] <= intervals[i][0]:
                heappop(heap)
              
            # add the new end time to the heap
            heappush(heap, intervals[i][1])
            
        return len(heap)

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
## 8. Maximum Subarray
- https://leetcode.com/problems/maximum-subarray/
- to find a **contiguous subarray**, Constantly check if the new element is bigger or the new ele + the localSum_soFar:
```py
def maxSubArray(nums):
  if len(nums) == 1:
      return nums[0]
      
  globalMax, localMax = nums[0], nums[0]
  
  for i in range(1, len(nums)):
      localMax = max(nums[i], localMax + nums[i])
      globalMax = max(localMax, globalMax)
      
  return globalMax
```

## 9. Best Time to Buy and Sell Stock
- https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
- Brute Force O(n2): iterate and for each node, iterate the rest of array and calculate the max profit.
- O(n), in one pass: we look at the trend of numbers in the arry, we want to keep track of the min value === lowest price and also the max point === max profit (point - lowest price):

```py
def maxProfit(prices):
    profit = 0
    if len(prices) == 1:
        return profit

    minPrice = float('Inf')
    maxProfit = 0
    
    for i in range(len(prices)):
        if prices[i] < minPrice:
            minPrice = prices[i]
            
        elif prices[i] - minPrice > maxProfit:
            maxProfit = prices[i] - minPrice
            
    return maxProfit
```
## 10. Best Time to Buy and Sell Stock II
- https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/solution/
```py
def maxProfit(self, prices):
      """
      :type prices: List[int]
      :rtype: int
      """
      profit = 0
      if len(prices) == 1:
          return profit

      profit = 0
      for i in range(1, len(prices)):
          diff = prices[i] - prices[i-1]
          if diff > 0:
              profit += diff

      return profit

```
## 11. Subtree of Another Tree
- https://leetcode.com/problems/subtree-of-another-tree/
- the condition of two trees being identical is their root values are the same and recursively valid for their left and right subtrees. Now if both subtree are None, their identical but if one is None and the other is not then it is false: `if not(t1 and t2): return t1 is t2`
- we use that helper in the main method to see if the whole root and subtree are identical if not then we recursively call the method for each left and right subtrees of the main tree.

```py
def isSubtree(self, root, subRoot):
        """
        :type root: TreeNode
        :type subRoot: TreeNode
        :rtype: bool
        """
        if self.isIdentical(root, subRoot): return True
        if root is None: return False
        
        return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)
        
    def isIdentical(self, t1, t2):
        if not (t1 and t2):
            return t1 is t2
        
        return (t1.val == t2.val and
               self.isIdentical(t1.left, t2.left) and 
               self.isIdentical(t1.right, t2.right))
```
## 12. Rotting Oranges:
- https://leetcode.com/problems/rotting-oranges/
- we need to find min elapse time meaning that everywhere in the matrix that we have a rotten one (2) they rot their neighboring nodes at the same time.
- Edge case: where there is no fresh orange, 


```py

import collections
class Solution(object):
    def __init__(self):
        self.visited = set()
        
    def orangesRotting(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        rows = len(grid)
        cols = len(grid[0])
        ones = set()
        twos = collections.deque()
        lapse = 0
        
        
        for i in range(rows):
            for j in range(cols):
                node = grid[i][j]
                if node == 1:
                    ones.add((i,j))
                    
                if node == 2:
                    twos.append((i,j))
       
        if len(ones) == 0: return 0
        
        lapse = self.bfs(twos, grid)
        
        
        for ele in ones:
            if ele not in self.visited:
                return -1
            
        return lapse
    
    
    def bfs(self, queue, grid):
        timeLapse = -1
        
        while queue:
            size = len(queue)
            timeLapse += 1
            
            for idx in range(size):
                i, j = queue.popleft()
                if (i, j) in self.visited: continue
                    
                self.visited.add((i,j))
                queue.extend(self.addNeighboringNodes(i,j, grid))
        return timeLapse
                
    
    def addNeighboringNodes(self, i, j, grid):
        result = []
        
        if i > 0 and grid[i-1][j] == 1 and (i-1, j) not in self.visited:
            grid[i-1][j] = 2
            result.append((i-1,j))
            
        if (j < len(grid[0]) - 1) and (grid[i][j+1] == 1) and (i,j+1) not in self.visited:
            grid[i][j+1] = 2
            result.append((i,j+1))
            
        if (i < len(grid) - 1) and grid[i+1][j] == 1 and (i+1, j) not in self.visited:
            grid[i+1][j] = 2
            result.append((i+1,j))
            
        if j > 0 and grid[i][j-1] == 1 and (i,j-1) not in self.visited:
            grid[i][j-1] = 2
            result.append((i,j-1))
            
        return result
```
## 13. Spiral Matrix II
- https://leetcode.com/problems/spiral-matrix-ii/
- We can walk in a spiral path and we can change direction when we go beyond the limit of our matrix or when the next element value is not zero. We then change direction. For direction, we can think of 4 direction where di, dj in `i += di, j += dj` are calculated like this: 
```
# "1": (0,1)
# "2": (1,0)
# "3": (0,-1)
# "4": (-1,0)
```
```py
def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """
        dir = [(0,1), (1,0), (0,-1), (-1,0)]
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        
        i, j = 0, 0
        dirVal = 0
        di, dj = dir[dirVal]
        
        for val in range(n*n):
            
            matrix[i][j] = val + 1
            
            # change direction
            nexti, nextj = i+di, j+dj
            
            if self._outOfBound(nexti, nextj, n) or (matrix[nexti][nextj] != 0):
                dirVal = (dirVal + 1)%4        
                
            di, dj = dir[dirVal]
                
                
            i += di
            j += dj
            
            
        return matrix
    
    def _outOfBound(self, i, j, n):
        return (i < 0) or (j < 0) or (i > n - 1) or (j > n - 1)
        
```
- O(n * n) time complexity
- O(1) no extra space is being used


## 14. House Robber
- https://leetcode.com/problems/house-robber/
- For each house the thief has two options: whether to rob it or not. if he robs that house[i] then he has to advance two before being able to rob but if he doesn't then we can advance to the next house. The outcome is the max of robbed values in those two choices. This is recursion with memoization: 
- At each step, the robber has two options. If he chooses to rob the current house, he will have to skip the next house on the list by moving two steps forward. If he chooses not to rob the current house, he can simply move on to the next house in the list. Let's see this mathematically.

`robFrom(i)=max( robFrom(i+1) , robFrom(i+2)+nums(i) )`

**- O(N) since we process at most N recursive calls, thanks to caching, and during each of these calls, we make an O(1) computation which is simply making two other recursive calls, finding their maximum, and populating the cache based on that.**

**- Space Complexity: O(N) which is occupied by the cache and also by the recursion stack.**


```py
class Solution(object):
    def __init__(self):
            self.memo = {}
            
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        return self.robFrom(0, nums)


    def robFrom(self, idx, nums):
        if idx > len(nums) - 1:
            return 0

        if idx not in self.memo:
            self.memo[idx] = max(self.robFrom(idx + 2, nums) + nums[idx], self.robFrom(idx + 1, nums))

        return self.memo[idx]

```

## 15. Maximum Area of a Piece of Cake After Horizontal and Vertical Cuts
- https://leetcode.com/problems/maximum-area-of-a-piece-of-cake-after-horizontal-and-vertical-cuts/
- Time complexity: `O(N log N + M logM)` where N and M are the size of vertical and horizontal cuts arrays.
```py
def maxArea(self, h, w, horizontalCuts, verticalCuts):
      """
      :type h: int
      :type w: int
      :type horizontalCuts: List[int]
      :type verticalCuts: List[int]
      :rtype: int
      """
      horizontalCuts.sort()
      verticalCuts.sort()

      maxHeight = max(horizontalCuts[0], h - horizontalCuts[-1])

      for i in range(1, len(horizontalCuts)): 
          maxHeight = max(horizontalCuts[i] - horizontalCuts[i-1], maxHeight)


      maxWidth = max(verticalCuts[0], w - verticalCuts[-1])
      for i in range(1, len(verticalCuts)):
          maxWidth = max(verticalCuts[i] - verticalCuts[i-1], maxWidth)

      return (maxWidth * maxHeight) % (10**9 + 7)
      
      
       # OR
      horizontalCuts.sort()
      verticalCuts.sort()

      horizontalCuts.append(h)
      verticalCuts.append(w)

      prev = 0
      maxWidth = 0
      for hCut in horizontalCuts:
          maxWidth = max((hCut - prev), maxWidth)
          prev = hCut

      prev = 0
      maxHeight = 0
      for wCut in verticalCuts:
          maxHeight = max((wCut - prev), maxHeight)
          prev = wCut

      return (maxWidth * maxHeight) % (10**9 + 7)
        
```
## 16. Reorder Data in Log Files
- https://leetcode.com/problems/reorder-data-in-log-files/
- Note that in python `sort` and `sorted` use **Timesort** algorithm which is a hybrid algorithm derived from **insertion sort** and **merge sort**. Time complexity **O(n log)** & worst space complexity **O(N) assuming each element is O(1) or O(M .N) if each element is M lenght**.
- **`ss.isalpha(), ss.isnumeric()` to check if its a letter or number**
- If we want to sort an array based on multiple criteria (for example first sort by name and if two entries have the same name, then sort by age and then by job), we use a tuple of keys: `sorted(key = (name, age, job))`.
- To first order letter ids and then digita ones, we use the first key like `(0, rest, _id) < (1,)`

- **Time complexity O(M . N log N)**
- Let **N** be the number of logs in the list and **M** be the maximum length of a single log.
    - **Time Complexity:** **O(M⋅N⋅logN)**
    - The sorted() in Python is implemented with the Timsort algorithm whose time complexity is **O(N⋅logN).**
    - Since the keys of the elements are basically the logs itself, the comparison between two keys can take up to **O(M) time.**

Therefore, **the overall time complexity of the algorithm is O(M⋅N⋅logN).**

- **Space Complexity: O(M⋅N)**
  - First, we need **O(M⋅N) space to keep the keys for the log.**
  - In addition, the **worst space complexity of the Timsort algorithm is O(N), assuming that the space for each element is O(1). Hence we would need O(M⋅N) space to hold the intermediate values for sorting.**

In total, the overall space complexity of the algorithm is \mathcal{O}(M \cdot N + M \cdot N) = \mathcal{O}(M \cdot N)O(M⋅N+M⋅N)=O(M⋅N).
```py
 def reorderLogFiles(self, logs: List[str]) -> List[str]:
        
      def getKey(log):
          _id, rest = log.split(' ', maxsplit=1)
          re = (0, rest, _id) if rest[0].isalpha() else (1,)
          print(re)
          return re


      return sorted(logs, key=getKey)
```
- If you're asked to write a comparator similar to java for comparison logs, you need to import `cmp_to_key` from `functools` to **convert a comparator function to key**:
```py
from functools import cmp_to_key

def reorderLogFiles(logs: List[str]) -> List[str]:
    c = CustomComparator()
        
    return sorted(logs, key=cmp_to_key(c.compare))

class CustomComparator(object):
    
    def comparator(self, t1, t2):
        if t1 < t2:
            return -1
        elif t1 > t2:
            return 1
        else:
            return 0
        
    def compare(self, log1, log2):
        split1 = log1.split(' ', maxsplit=1) #(id, rest)
        split2 = log2.split(' ', maxsplit=1)
        
        isDigit1 = split1[1][0].isnumeric()
        isDigit2 = split2[1][0].isnumeric()
        
        # case 1: two logs are letters
        if (not isDigit1 and not isDigit2):
            cmp = self.comparator(split1[1], split2[1])
            
            if cmp != 0:
                return cmp
            else:
                return self.comparator(split1[0], split2[0])
        
        # case 2: one log is letter and the other one is digit
        if (not isDigit1 and isDigit2):
            return -1
        elif (not isDigit2 and isDigit1):
            return 1
        else: # case 3: two logs are digits
            return 0
```
## 17. Analyze User Website Visit Pattern

- https://leetcode.com/problems/analyze-user-website-visit-pattern/
- `zip` and `*`: zip takes iterables (zero or more) and aggregates them in a tuple and return it. The zip()function returns an iterator of tuples based on the iterable objects.

    - If we do not pass any parameter, zip() returns an empty iterator
    - If a single iterable is passed, zip() returns an iterator of tuples with each tuple having only one element.
    - If multiple iterables are passed, zip() returns an iterator of tuples with each tuple having elements from all the iterables.

    - Suppose, two iterables are passed to zip(); one iterable containing three and other containing five elements. Then, the returned iterator will contain three tuples. It's because iterator stops when the shortest iterable is exhausted.
```py
number_list = [1, 2, 3]
str_list = ['one', 'two', 'three']

# No iterables are passed
result = zip()

# Converting iterator to list
result_list = list(result)
print(result_list)   #[]

# Two iterables are passed
result = zip(number_list, str_list)

# Converting iterator to set
result_set = set(result)
print(result_set) # {(2, 'two'), (3, 'three'), (1, 'one')}


numbersList = [1, 2, 3]
str_list = ['one', 'two']
numbers_tuple = ('ONE', 'TWO', 'THREE', 'FOUR')

# Notice, the size of numbersList and numbers_tuple is different
result = zip(numbersList, numbers_tuple)

# Converting to set
result_set = set(result)
print(result_set)  # {(2, 'TWO'), (3, 'THREE'), (1, 'ONE')}

result = zip(numbersList, str_list, numbers_tuple)

# Converting to set
result_set = set(result)
print(result_set)  # {(2, 'two', 'TWO'), (1, 'one', 'ONE')}
```

- The * operator can be used in conjunction with zip() to unzip the list. `zip(*zippedList)`:
```py
coordinate = ['x', 'y', 'z']
value = [3, 4, 5]

result = zip(coordinate, value)
result_list = list(result)
print(result_list)   # [('x', 3), ('y', 4), ('z', 5)]

c, v =  zip(*result_list)
print('c =', c)  # c = ('x', 'y', 'z')
print('v =', v)  # v = (3,4,5)
```

- **Python combination and permutation**:
- for a given input if we should not change the place of elements in that input, use combination (n choose k: n!/(n-k)! k!):
```py
cc = combonations('ABC', 2)  # returns an iterable
for ele in cc:
  print(cc)  # returns 2-length tuple in sorted order
  
('A', 'B')
('A', 'C')
('B', 'C')
```
- if we can change the order, user permutation (n permutes k : n!/(n-k)!):
```py
cc = permutations('ABC', 2)  # returns an iterable
for ele in cc:
  print(cc)  # returns 2-length tuples all possible ordering
  
('A', 'B')
('A', 'C')
('B', 'A')
('B', 'C')
('C', 'A')
('C', 'B')
```
- For a list of username, timestamp and websites if we want to extract all websites visited by each users in chronological order, we can use zip to create an interatble consists of tuples of 3 elements and then order based on username and then timestamp so we can a list of websites for each user based on the time order:
```py
m = sorted(zip(username, timestamp, website), key= lambda x:(x[0], x[1]))
```

-  `sorted(myDict)` ona dictionary without any key returns an array of dict keys sorted in **ascending order**.
- description: https://leetcode.com/problems/analyze-user-website-visit-pattern/discuss/899805/DETAILED-Easy-to-Follow-Python-Solution-%2B-Clearly-Explained
```py
def mostVisitedPattern(self, username: List[str], timestamp: List[int], website: List[str]) -> List[str]:
		
		# Create tuples as shown in description
		# The timestamps may not always be pre-ordered (one of the testcases)
		# Sort first based on user, then time (grouping by user)
		# This also helps to maintain order of websites visited in the later part of the solution
		
		userTable = {}
    combined = sorted(zip(username, timestamp, website), key=lambda x: (x[0], x[1]))

    for user, time, web in combined:
        if user in userTable:
            userTable[user].append(web)
        else:
            userTable[user] = [web]
            
     # create all possible 3-sequences combinations for each user
     # Get unique 3-sequence (note that website order will automatically be maintained)
		# Note that we take the set of each 3-sequence for each user as they may have repeats
		# For each 3-sequence, count number of users
     patternCount = {}
     
     for user, webs in userTable.items():
            usersCombinations = set(combinations(webs, 3))
            for pattern in usersCombinations:
                if pattern in patternCount:
                    patternCount[pattern] += 1
                else: 
                    patternCount[pattern] = 1
        
     print(patternCount)
     
    # Re-iterating above step for clarity
		# 1. first get all possible 3-sequences combinations(sites, 3)
		# 2. then, count each one once (set)
		# 3. finally, count the number of times we've seen the 3-sequence for every user (patterns.update(Counter)) 
		# - updating a dictionary will update the value for existing keys accordingly (int in this case)
    
    # get most frequent 3-sequence sorted lexicographically
     mm = sorted(patternCount) # sorted on dictionary returns a list of dict keys sorted in ascending order
     
     return max(mm, key=patternCount.get)
```
## 18. Sort Integers by The number of 1 Bits
- https://leetcode.com/problems/sort-integers-by-the-number-of-1-bits/
- A function to create a binary of an int in python is `bin(123)` which returns as **string** `0b1111011` and we can trim the first two letters by `bin(123)[2:]`.
```py
def sortByBits(self, arr):
	"""
	:type arr: List[int]
	:rtype: List[int]
	"""
	def getKey(num):
	    binary, count = self._getBinary(num)
	    return (count, num)

	return sorted(arr, key = getKey)


def _getBinary(self, num):
	result = 0
	count = 0
	n = 0

	while num:

	    resid = num % 2
	    if resid == 1: count += 1

	    result = resid * 10**n + result
	    n += 1
	    num = num // 2

	return (result, count)

```
## 19. Ones & Zeros
- https://leetcode.com/problems/ones-and-zeroes/
- Total number of subsets of a set is `2^(n) - 1`. So the Brute force solution will be time complexity of `O(m * 2^n)` where n is lenght of the set and m is the average lenght of the subset.
- Good explanaition: https://leetcode.com/problems/ones-and-zeroes/discuss/814077/Dedicated-to-Beginners
**- step1 : understand the problem**
	- we have been given a array of BinaryStrings and two numbers, viz no of zeros and no of ones.
	Our task is to tell how many strings can we make by using available no of zeros and ones, such that we are able to make the maximum no of strings present in given array.

**- step 2: logic building and deriving recurence relation**

**1st sub step : asking questions**

1. question that i should ask is what is the max no of strings can i make from given avaible choices.
( now assume you are at a particular index on the array , these are the questions that matter )
2. if i use this current string then what all choices i am left with ? what is the max possible ans for those choices.
3. if i dont use this current string then i have other options( other indexes ) to explore , whats going to be the answer for amoung these choices.
4. Finally : if i know the answer for both the paths then i ould be able to decide wether i should select this current string or not.
2nd sub step : Ariving at recurence relation

**2nd sub step : Ariving at recurence relation**
```py
at some Ith index my options is going to be 
1 . if i use decide to build this string then 
	 answerFor( next index , currAvailableZeros - currentStringZeros , currAvailableOnes - currentStringOnes )
2. if i dont decide to build this string ( assuming i will get better answer later, if i save available zeros and ones)
	 answerFor( next index , currAvailableZeros , currAvailableOnes  )
	 
ans_for_ith = maxOf ( if i build this current string , if i dont build )
```
- with memoization:
- Time complexity: **O(l * m * n)**: memo array with m and n sizes of zeros and ones and l is the lenght of strs array.
- Space complexity: **O(l * m * n)**: memo array with m and n sizes of zeros and ones and l is the lenght of strs array.

```py
class Solution(object):
    def __init__(self):
        # dp[idx:zeros:ones]
        self.dp = {}
        
    def findMaxForm(self, strs, m, n):
        """
        :type strs: List[str]
        :type m: int
        :type n: int
        :rtype: int
        """
        # create a list of tuples including #zeros and #ones for each string
        arr = []
        for ele in strs:
            arr.append((ele.count('0'), ele.count('1')))

        return self.helper(arr, 0, m, n)
    
    def helper(self, arr, idx, zeros, ones):

        if idx == len(arr) or (zeros == 0 and ones == 0):
            return 0
        
        key = str(idx) + ':' + str(zeros) + ':' + str(ones)
        
        if key in self.dp:
            return self.dp[key]
        
        if arr[idx][0] <= zeros and arr[idx][1] <= ones:
            currentZeros, currentOnes = arr[idx][0], arr[idx][1]
            include = self.helper(arr, idx + 1, zeros - currentZeros, ones - currentOnes) + 1
            exclude = self.helper(arr, idx + 1, zeros, ones)

            self.dp[key] = max(include, exclude)
                
        else:

            self.dp[key] = self.helper(arr, idx + 1, zeros, ones)
                
        return self.dp[key]

```
## 20. Minimum Cost to connect sticks
- https://leetcode.com/problems/minimum-cost-to-connect-sticks/
- We can use heap datastructure to always pop the two smallest values. We want to use the smallest sticks and use the longest ones at the end in a fewer times. Everytime we combine two sticks, the result is a larger one and we want to compare that against other sticks in the array to make sure that we always pick the shortest one for the next combine step.

- Time complexity: `O(N log N)`. Adding an array element to a heap takes `N log N` cause it needs to sort them for min heap. Space complexity is `O(N)`.

```py
import heapq

def connectSticks(sticks: List[int]) -> int:
	if len(sticks) == 1:
	    return 0

	h = []

	for stick in sticks:
	    heappush(h, stick)

	cost = 0

	while len(h) != 1:
	    ele1, ele2 = heappop(h), heappop(h)

	    sumEle = ele1 + ele2

	    cost += sumEle
	    # add the combined stick back to heap
	    heappush(h, sumEle)

	return cost
```
