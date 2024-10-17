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
- **Time complexity**: `O(N * M)` where M is the numner of rows and N is the number of columns.
- **Space complexity**: `O(N * M)`worst case where all the grid is island (1) where DFS goes M * N deep.

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

### Less sophisticated solution that still works with the concept above:
```py
class Solution(object):
    def cloneGraph(self, node):
        """
        :type node: Node
        :rtype: Node
        """
        if node is None:
            return None
        hashMap = {}
        start = node
        self.dfs(node, hashMap)
        
        for n in hashMap:
            nn = hashMap[n]
            for nei in n.neighbors:
                nn.neighbors.append(hashMap[nei])
        
        return hashMap[start]

    def dfs(self, node, hashMap):
        if node:
            if node not in hashMap:
                hashMap[node] = Node(node.val)
                
            for neigh in node.neighbors:
                if neigh not in hashMap:
                    self.dfs(neigh, hashMap)
```

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
- **Time complexity**: `O(N + M)` where N is a number of node and M is a number of edges
- **Space complexity**: `O(N)`. The space occupied by the `visited` hash map **and in addition to that, space would also be occupied by the queue since we are adopting the BFS approach here. The space occupied by the queue would be equal to O(W) where W is the width of the graph. Overall, the space complexity would be O(N).**

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

```py
# first approach which results in Time limit exceeds
def __init__(self):
        self.count = 0
    def findTargetSumWays(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        if len(nums) == 0 or target is None:
            return 0
        
        if len(nums) == 1:
            if abs(target) == abs(nums[0]):
                return 1
            else:
                return 0

        def backtrack(currVal, idx):
            if idx == len(nums):
                if currVal == target:
                    self.count += 1
                return
            
            backtrack(currVal + nums[idx], idx+1)
            backtrack(currVal - nums[idx], idx+1)

        backtrack(0, 0)

        return self.count

# second approach with memoization
class Solution(object):
    def findTargetSumWays(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        if len(nums) == 0 or target is None:
            return 0
        
        if len(nums) == 1:
            if abs(target) == abs(nums[0]):
                return 1
            else:
                return 0

        memo = {}
        def backtrack(currVal, idx, memo):
            if idx == len(nums):
                if currVal == target:
                    return 1
                return 0
            
            if (currVal, idx) in memo:
                return memo[(currVal, idx)]
            
            count1 = backtrack(currVal + nums[idx], idx+1, memo)
            count2 = backtrack(currVal - nums[idx], idx+1, memo)

            memo[(currVal, idx)] = count1 + count2

            return memo[(currVal, idx)]

        count = dfs(0, 0, memo)

        return count
```

## Number of connected components in an Undirected Graph
- https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph
```py
class Solution(object):
    def countComponents(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: int
        """
        if len(edges) == 0:
            return n

        if n <= 1:
            return n

        connectivity = defaultdict(list)
        for edge in edges:
            connectivity[edge[0]].append(edge[1])
            connectivity[edge[1]].append(edge[0])

        visited = set()
        count = 0
        for node in connectivity:
            if node not in visited:
                self.dfs(node, visited, connectivity)
                count += 1

        return count + (n - len(connectivity))


    def dfs(self, node, visited, connectivity):
        if node != None:
            if node not in visited:
                visited.add(node)
                for nextNode in connectivity[node]:
                    self.dfs(nextNode, visited, connectivity)


        return visited
```

## Topological Order/Sort for DAG (Directed Acyclic Graphs)
- https://leetcode.com/problems/course-schedule/?envType=company&envId=amazon&favoriteSlug=amazon-thirty-days
- https://www.youtube.com/watch?v=cIBFEhD77b4
- many real world situations can be modeled as a graph with directed esges, where some events must occur before others:
  - program build dependencies
  - college class pre-requisites
  - Event scheduling
- A topological ordering is an ordering of the nodes in a directed graph where for each directed esge from node A to node B, node A appears before node B in the ordering.
- Kahn's algorithm is a simple topological sort algorithm can find a topological ordering in `O(E + V)` time.
- Topological orderdings are **NOT unique.**
- ONly certain types of graphs have a topological orderings. These are DAGs. A DAG is a finite directed graph with no directed cycles.
- **Kahn's algorithm:** an intuition behind Kahn's algorithm is to repeatedely remove nodes without any dependencies from the graph and add them to the topological ordering. As nodes without (incoming degree of the node is zero)depenedencies (and their outgoing edges) are removed from the graph, new nodes without dependencies should become free. We repeat removing nodes without dependencies from the graph untill all nodes are processed or a cycle is discovered.
```py
if len(prerequisites) == 0:
            return True

        outGress = defaultdict(int)
        inGress = [0] * numCourses

        for node in prerequisites:
            start, end = node[0], node[1]
            inGress[end] += 1
            if start in outGress:
                outGress[start].append(end)
            else:
                outGress[start] = [end]

        queue = collections.deque([])
        for k, v in enumerate(inGress):
            if v == 0:
                queue.append(k)

        inOrder = []
        while queue:
            node = queue.popleft()
            inOrder.append(node)
            if node in outGress:
                for child in outGress[node]:
                    inGress[child] -= 1
                    if inGress[child] == 0:
                        queue.append(child)
        
        return len(inOrder) == numCourses
```
**A more efficient Runtime and space:**
```py
if len(prerequisites) == 0:
            return True
        
        ingressMap = defaultdict(int)
        outgressMap = defaultdict(list)
        
        for pre in prerequisites:
            ingressMap[pre[1]] = ingressMap.get(pre[1], 0) + 1
            if pre[0] in outgressMap:
                outgressMap[pre[0]].append(pre[1])
            else:
                outgressMap[pre[0]] = [pre[1]]
            
        
        
        queue = collections.deque()
        for course in range(0, numCourses):
            ingress = ingressMap.get(course, 0)
            if ingress == 0:
                queue.append(course)
        
        count = 0
        while queue:
            course = queue.popleft()
            count += 1
            if course in outgressMap and len(outgressMap[course]) > 0:
                for c in outgressMap[course]:
                    if c in ingressMap:
                        ingressMap[c] -= 1
                        if ingressMap[c] == 0:
                            queue.append(c)
            
        return count == numCourses
```
