### Array of Doubled Pairs
- https://leetcode.com/problems/array-of-doubled-pairs/
- for each element in array, x, we need to find whether `2*x or x/2` exist. However, if we sort the array based on their abs value, then we need to only check for the existence of `2*x` because, x is the least value and so `x/2` can not exist.
- We might have double or more than occurance of numbers so we want to keep count of values we visited and remove them from the count so we don't use the same value twice. FOr that we need a hashmap of values and their counts.
- The time complexity is `O(NlogN)`. `N` is for creating a hashmap of values and their counts and even though we iterate on the sorted array, we only visit half of it each time because we remove the two visited ones. So if the lenght was 8, next time 6 .... Every time, we're going to look at half of the values and map the rest with its occurances


# Breadth First Search (BFS)

- One common application of Breadth-first Search (BFS) is to find the shortest path from the root node to the target node. BFS of a graph is similar to BFS of a tree. The only catch is, unlike tree, graphs may contain cycles. so we may come to th same node.To avoid processing a node more than once, we use a boolean visited array. 
- https://www.programiz.com/dsa/graph-bfs
- If a node X is added to the queue in the kth round, the length of the shortest path between the root node and X is exactly k. That is to say, you are already in the shortest path the first time you find the target node.
- The time complexity in a graph is O(V+E), where V: number of vertices and E: number of edges.

```py
# BFS algorithm in Python
import collections

def bfs(graph, root):
    queue = collections.deque([root])
    visited = set()
    depth = -1
    
    while queue:
        size = len(queue)
        depth += 1
        
        for _ in range(size):
            cur = queue.popleft()
            if cur in visited: continue
            visited.add(cur)
            queue.extend(nextLayer(node))
        
def nextLayer(node):
    listNodes = []
    // add all successors of node to listNodes
    return listNodes
    
            
if __name__ == '__main__':
    graph = {0: [1, 2], 1: [2], 2: [3], 3: [1, 2]}
    print("Following is Breadth First Traversal: ")
    bfs(graph, 0)
```

# Depth First Search (DFS)

- We don't know if the found path is the shortest path between two vertices.
- Instead of queue in BFS, we use stack (LIFO) in DFS.
- The average time complexity for DFS on a graph is O(V + E), where V is the number of vertices and E is the number of edges. In case of DFS on a tree, the time complexity is O(V), where V is the number of nodes.
- We say average time complexity because a set’s `in` operation has an average time complexity of O(1). If we used a list, the complexity would be higher.
- What if you want to find the shortest path?
Hint: Add one more parameter to indicate the shortest path you have already found.

```py
visited = set() # Set to keep track of visited nodes.
def dfs(visited, graph, node):
    if node not in visited:
        print (node)
        visited.add(node)
        for neighbour in graph[node]:
            dfs(visited, graph, neighbour)
```

```py
# DFS algorithm
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)

    print(start)

    for next in graph[start] - visited:
        dfs(graph, next, visited)
    return visited


graph = {'0': set(['1', '2']),
         '1': set(['0', '3', '4']),
         '2': set(['0']),
         '3': set(['1']),
         '4': set(['2', '3'])}

dfs(graph, '0')
```

# DFS with stack:
- Given the root of a binary tree and an integer targetSum, return true if the tree has a root-to-leaf path such that adding up all the values along the path equals targetSum.

**DFS with Stack**
```py
def hasPathSum(root, targetSum):
    if not root:
        return False

    sumSofar = root.val
    stack = [(root, 0)]

    while stack:
        print([(node.val, sumAll) for node, sumAll in stack])
        node, localSum = stack.pop()

        currentSum = node.val + localSum

        if not(node.left or node.right):
            if currentSum == targetSum:
                return True

        else:
            if node.left: stack.append((node.left, currentSum))
            if node.right: stack.append((node.right, currentSum))
                
    return False
```
```py
# 1. DFS with stack 2
def hasPathSum(root, targetSum):
    if not root:
        return False
        
    stack = [(root, root.val)]
    
    while stack:
        node, val = stack.pop()
        
        if val == targetSum and not(node.left or node.right):
            return True
            
        if node.left: stack.append((node.left, val + node.left.val))
        if node.right: stack.append((node.right, val + node.right.val))
    return False
```

**Recursive**
```py
def hasPathSum(root, targetSum):
    if not root:
        return False
        
    if root.val == targetSum and not(root.left, root.right):
        return True
        
    return hasPathSum(root.left, targetSum - root.val) or hasPathSum(root.right, targetSum - root.val)
```
**BFS with queue**
```py
def hasPathSum(root, targetSum):
    if not root:
        return False
        
    queue = [(root, targetSum - root.val)]
    
    while queue:
        curr, val = queue.popleft()
        if val == 0 and not(curr.left or curr.right):
            return True
        if curr.left: queue.append((curr.left, val - curr.left.val))
        if curr.right: queue.append((curr.right, val - curr.right.val))
    return False

```

### Path Sum II: https://leetcode.com/problems/path-sum-ii/
**DFS with stack**
```py
def pathSum(self, root, targetSum):
        """
        :type root: TreeNode
        :type targetSum: int
        :rtype: List[List[int]]
        """
        result = []
        if not root:
            return result

        stack = [(root, 0, [])]


        while stack:
            node, localSum, branchList = stack.pop()
            localSum += node.val

            if localSum == targetSum and not(node.left or node.right):
                result.append(branchList + [node.val])

            else:
                if node.right:
                    stack.append((node.right, localSum, branchList + [node.val]))
                if node.left:
                    stack.append((node.left, localSum, branchList + [node.val]))

        return result
```
```py
def pathSum(self, root, targetSum):
        """
        :type root: TreeNode
        :type targetSum: int
        :rtype: List[List[int]]
        """
        result = []
        if not root:
            return result
        
        stack = [(root, targetSum, [root.val])]      
        
        while stack:
            node, sumNodes, branchResult = stack.pop()
                  
            # reaching a leaf node
            if not(node.left or node.right) and node.val == sumNodes:
                result.append(branchResult)    
                
            if node.left: stack.append((node.left, sumNodes - node.val, branchResult+ [node.left.val]))
            if node.right: stack.append((node.right, sumNodes - node.val, branchResult+[node.right.val]))

        return result

```
**Recursive DFS**
```py
def pathSum(self, root, targetSum):
        """
        :type root: TreeNode
        :type targetSum: int
        :rtype: List[List[int]]
        """
        return helper(root, targetSum, [], [])
    
def helper(node, targetSum, branchList, result):
    if node == None:
        return []
    
    if node.val == targetSum and not(node.left or node.right):
        newBranch = branchList + [node.val]
        result.append(newBranch)
        
    if node.left: helper(node.left, targetSum - node.val, branchList + [node.val], result) 
    if node.right: helper(node.right, targetSum - node.val, branchList + [node.val], result)
    
    return result
```
```py
def pathSum(self, root, targetSum):
        """
        :type root: TreeNode
        :type targetSum: int
        :rtype: List[List[int]]
        """
        # Recursive DFS
        if not root:
            return []
        result = []
        self.dfs(root, targetSum, [root.val], result)
        return result
    
    def dfs(self, node, sumNodes, branchResult, result):
        
        if not(node.left or node.right) and sumNodes == node.val:
            result.append(branchResult)
            
        if node.left: self.dfs(node.left, sumNodes - node.val, branchResult + [node.left.val], result)     
        if node.right: self.dfs(node.right, sumNodes - node.val,  branchResult + [node.right.val], result)

```


## Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it is able to trap after raining.
# Example:
# Input: [0,1,0,2,1,0,1,3,2,1,2,1]
# Output: 6

- The brute force solution: for each location, find the max on the left and max on the right of that location and take the min of them
- Optimal solution: calculate the column of water in each location based on the followings:
    - If there was an infinite tall wall on the right end of the array, the water in each location would be the height of max so far on the left of the location - height of the location
    - Now for the locations on the right of the infinite wall, the water in each location coming from right to left is heigth of max so far on the right of the location - height of location.

```py
def trap(self, height):
    """
    :type height: List[int]
    :rtype: int
    """
    size = len(height)
    area = 0

    if size == 0:
        return area

    globalIdx, globalMax = self.findGlobalMax(height)

    # water trap values coming from left to right: find the local max on the left side and calculate the trapped water = (local_max - current_height)*width(=1)
    max_height_local = height[0]
    for i in range(0, globalIdx):
        if height[i] > max_height_local:
            max_height_local = height[i]

        area += (max_height_local - height[i]) * 1

    # water trap values coming from right to left: find the local max on the right side and calculate trapped water = (local_max - current_height)*width
    max_height_local = height[-1]
    for i in range(size - 1, globalIdx, -1):
        if height[i] > max_height_local:
            max_height_local = height[i]
        area += (max_height_local - height[i]) * 1

    return area

def findGlobalMax(self, height):
    maxIdx, maxHeight = 0, height[0]

    for i, val in enumerate(height):
        if val > maxHeight:
            maxIdx = i
            maxHeight = val
    return (maxIdx, maxHeight)
 ```

## Longest consecutive sequence:
- https://leetcode.com/problems/longest-consecutive-sequence/
- **solutions:**
- **1. Brute force**:
- it just considers each number in nums, attempting to count as high as possible from that number using only numbers in nums. After it counts too high (i.e. currentNum refers to a number that nums does not contain), it records the length of the sequence if it is larger than the current best. The algorithm is necessarily optimal because it explores every possibility.
```py
def longestConsecutive(nums):
    if not nums:
        return 0

    longest = 0
    for num in nums:
        curr = num
        local_streak = 1
        
        while curr + 1 in nums:
            local_streak += 1
            curr += 1
        
        longest = max(longest, local_streak)
    return longest
```
- **Time complexity**: O(n^3) - the for loop: n, the while loop is n and the `in` operator in `while` is O(n)..
- **Space complexity** O(1).

**2. Define nums as a set so we do lookup in in O(1) time:**
- First turn the input into a set of numbers. That takes O(n) and then we can ask in O(1) whether we have a certain number. Then go through the numbers. If the number x is the start of a streak (i.e., x-1 is not in the set), then test y = x+1, x+2, x+3, ... and stop at the first number y not in the set. The length of the streak is then simply y-x and we update our global best with that. Since we check each streak only once, this is overall **O(n)**.
```py
def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums = set(nums)
        result = 0
        
        for x in nums:
            if x - 1 not in nums:
                y = x + 1
                while y in nums:
                    y += 1
                result = max(result, y - x)
        return result
```
## Heap data structure (min heap, max heap and Priority Queue): https://www.programiz.com/dsa/priority-queue
- Python heap: `heappush, heappop...` for min heap

## Spiral Array
- https://leetcode.com/problems/spiral-matrix/

```py
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        rows = len(matrix)
        cols = len(matrix[0])
        
        output = []
        
        minRow = minCol = 0
        maxRow = rows - 1
        maxCol = cols - 1
        
        while len(output) < rows * cols:
            
            if minRow <= maxRow and minCol <= maxCol:
                i = minRow
                for j in range(minCol, maxCol + 1):
                    output.append(matrix[i][j])
                minRow += 1

                # if minRow > maxRow:
                #     break

            if minRow <= maxRow and minCol <= maxCol:
                j = maxCol
                for i in range(minRow, maxRow + 1):
                    output.append(matrix[i][j])
                maxCol -= 1

                # if minCol > maxCol:
                #     break

            if minRow <= maxRow and minCol <= maxCol:
                i = maxRow
                for j in range(maxCol, minCol - 1, -1):
                    output.append(matrix[i][j])
                maxRow -= 1

                # if minRow > maxRow:
                #     break

            if minRow <= maxRow and minCol <= maxCol:
                j = minCol
                for i in range(maxRow, minRow - 1, -1):
                    output.append(matrix[i][j])
                minCol += 1

                # if minCol > maxCol:
                #     break
            
        
        return output
```
## Pascal Triangle
- https://leetcode.com/problems/pascals-triangle/

- our output list will store each row as a sublist
- the first and last element of each sublist is 1.
- we then can calculate each element in between based on pervious sublist elements
```py
def generate(numRows):
    triangle = [[1]]
    
    if numRows > 1:
        for row in range(1, numRows):
            sublist = [0] * (row + 1)
            sublist[0] = sublist[-1] = 1
            
            for k in range(1, row):
                sublist[k] = triangle[row - 1][k] + triangle[row - 1][k - 1]
            
            triangle.append(sublist)

    
    return triangle
```

## Minimum Size Subarray Sum
- https://leetcode.com/problems/minimum-size-subarray-sum/
**Algorithm**
1. Initialize `left` pointer to 0
2. Iterate over the array:
    - Add to the sum
    - while sum is larger than the target:
        - update the answer
        - remove from the sum index....
```py
def minSubarray(nums, target):
    maxVal = max(nums)
    if maxVal >= target:
        return 1
    
    if len(nums) == 0:
        return 0
    
    left = 0
    sumSoFar = 0
    countSoFar = 0
    minCount = float('Inf')
    
    for i in range(len(nums)):
        sumSoFar += nums[i]
        while sumSoFar >= target:
            minCount = min(minCount, i - left + 1)
            sumSoFar -= nums[left]
            left += 1
            
    if minCount != float('Inf'):
        return minCount
    else:
        return 0
```
