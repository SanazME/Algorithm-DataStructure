## DFS & BFS
- https://github.com/SanazME/Algorithm-DataStructure/tree/master/array_stack_queues

### DFS
- Instead of **queue in BFS**, we use **stack in DFS (LIFO)**.
- we can use DFS for pre-order, in-order and post-order traversal in trees.
- **we never trace back unless we reach the deepest node**. Unlike BFS where **we never go deeper unless it has already visited all nodes at the current level**.
- **BFS finds the shortest path from the root node to the target node** but not in DFS.
- We typically use **recursion** for DFS. We can also have DFS without recursion with **stack**.
- The average time complexity for DFS on a graph is O(V + E), where V is the number of vertices and E is the number of edges. In case of DFS on a tree, the time complexity is O(V), where V is the number of nodes.

- **In DFS with recursion never forget the implicit call stack (system stack / execution stack) when calculating space complexity**
- **The size of call stack is the depth of DFS**.

#### Problem 1: https://leetcode.com/problems/number-of-islands/
solution: 
- Treat the 2d grid map as an undirected graph and there is an edge between two horizontally or vertically adjacent nodes of value '1'.
**Algorithm**:
- Linear scan the 2d grid map, if a node contains a '1', then it is a root node that triggers a Depth First Search. During DFS, every visited node should be set as '0' to mark as visited node. Count the number of root nodes that trigger DFS, this number would be the number of islands since each DFS starting at some root identifies an island.
```py
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if grid is None or len(grid) == 0:
            return 0
        
        rows = len(grid)
        cols = len(grid[0])
        num_islands = 0
        
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == '1':
                    num_islands += 1
                    self.dfs(i, j, grid)
        
        return num_islands
        
        
    def dfs(self, i, j, grid):
        rows = len(grid)
        cols = len(grid[0])
        
        if i < 0 or i >= rows or j < 0 or j >= cols or grid[i][j] == '0':
            return
        
        grid[i][j] = '0'
        
        self.dfs(i - 1, j, grid)
        self.dfs(i + 1, j, grid)
        self.dfs(i, j - 1, grid)
        self.dfs(i, j + 1, grid)
```

#### Problem 2:
- https://leetcode.com/problems/clone-graph/

**1. DFS**

- The basic intuition for this problem is to just copy as we go. We need to understand that we are dealing with a graph and this means a node could have any number of neighbors. This is why neighbors is a list. What is also crucial to understand is that we don't want to get stuck in a cycle while we are traversing the graph. According to the problem statement, any given undirected edge could be represented as two directional edges. So, if there is an undirected edge between node A and node B, the graph representation for it would have a directed edge from A to B and another from B to A. After all, an undirected graph is a set of nodes that are connected together, where all the edges are bidirectional. 

- To avoid getting stuck in a loop we would need some way to keep track of the nodes which have already been copied. By doing this we don't end up traversing them again.

- **our helper function takes in a node from origianl graph as input and returns copied node of that node**.
- **node is a data structure defined that has a value and a list of neighbor nodes.**

**Algorithm**
1. Start traversing the graph from the given node.
2. We would take a hash map to store the reference of the copy of all the nodes that have already been visited. The `key` for the hash map would be the node of the original graph and corresponding `value` would be the corresponding cloned node of the cloned graph. If the node already exists in the visited we return corresponding stored reference of the cloned node.
3. If we don't find the node in the `visited` hash map, we create a copy of it and put it in the hash map.
4. Now make the recursive call for the neighbors of the node. **Pay attention to how many recursion calls we will be making for any given node. For a given node the number of recursive calls would be equal to the number of its neighbors. Each recursive call made would return the clone of a neighbor**. We will prepare the list of these clones returned and put into neighbors of clone node which we had created earlier. This way we will have cloned the given node and it's neighbors.

```py
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val, neighbors):
        self.val = val
        self.neighbors = neighbors
"""
class Solution(object):

    def __init__(self):
        # Dictionary to save the visited node and it's respective clone
        # as key and value respectively. This helps to avoid cycles.
        self.visited = {}

    def cloneGraph(self, node):
        """
        :type node: Node
        :rtype: Node
        """
        if not node:
            return node

        # If the node was already visited before.
        # Return the clone from the visited dictionary.
        if node in self.visited:
            return self.visited[node]

        # Create a clone for the given node.
        # Note that we don't have cloned neighbors as of now, hence [].
        clone_node = Node(node.val, [])

        # The key is original node and value being the clone node.
        self.visited[node] = clone_node

        # Iterate through the neighbors to generate their clones
        # and prepare a list of cloned neighbors to be added to the cloned node.
        if node.neighbors:
            clone_node.neighbors = [self.cloneGraph(n) for n in node.neighbors]

        return clone_node
```
- **Time complexity**: `O(N + M)` where N is a number of node and M is a number of edges
- **Space complexity**: `O(N)`. The space occupied by the `visited` hash map **in addition to that, space would also be occupied by the recursion stack since we are adopting a recursive approach here. The space occupied by the recursion stack would be equal to O(H) where H is the height of the graph. Overall, the space complexity would be O(N).**


**2. BFS**
- If the recursion stack is what we are worried about then DFS is not our best bet. We can use BFS instead.
```py
"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""
import collections

class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        
        if not node:
            return node
        
        visited = {}
        
        queue = collections.deque([node])
        visited[node] = Node(node.val, []) 
        
        while queue:
            curr = queue.popleft()
            
            for neighbor in curr.neighbors:
                if neighbor not in visited:
                    visited[neighbor] = Node(neighbor.val, [])
                    queue.append(neighbor)
                    
                visited[curr].neighbors.append(visited[neighbor])
                
        return visited[node]
```

#### Problem 2:
- https://leetcode.com/problems/target-sum/
- At first I just remember the current index and current target, and for each index, either subtract the nums[i] from S or add it to S. But this got TLE, them I came up with this solution. Just store the intermediate result with (index, s) and this got accepted.
```py
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        
        if nums is None or target is None:
            return 0
        
        def dfs(idx, currTarget):
            if (idx, currTarget) in cache:
                return cache[(idx, currTarget)]
            else:
                ways = 0
                if idx == len(nums):
                    if currTarget == 0:
                        ways += 1
                else:
                    ways = dfs(idx + 1, currTarget - nums[idx]) + dfs(idx + 1, currTarget + nums[idx])
                    
            
                cache[(idx, currTarget)] = ways
                
                return cache[(idx, currTarget)]
        
        
        cache = {}
        return dfs(0, target)
```
**Time complexity**:
the complexity is `O(n * s)`, where n is the length of the input and s the target. The reason is that each entry in our cache table corresponds to each node in the recursion tree of the above algorithm. O(ns) in our case will most likely be less than O(2^n) because usually there are duplicate nodes (duplicate meaning "at the same height of the recursion tree and with the same residual sum"), but there are cases (rare) in which the two are the same, therefore we don't exceed time limit.


## Number of connected components in an Undirected Graph
- https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph
