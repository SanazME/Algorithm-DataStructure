## Feature #1: Find all the people on Facebook that are in a user’s friend circle.
- https://leetcode.com/problems/number-of-islands/
- https://leetcode.com/problems/number-of-islands-ii/
- https://www.cs.princeton.edu/courses/archive/spring19/cos226/lectures/15UnionFind.pdf
- https://algs4.cs.princeton.edu/15uf/

Solution: We can think of the symmetric input matrix as an **undirected graph**.  We can treat our input matrix as an **adjacency matrix**; our task is to find the number of connected components.

Solution: [Number of islands](https://github.com/SanazME/Algorithm-DataStructure/blob/master/AmazonQuestions.md#5-number-of-islands)
- Time complexity:
  - DFS, BFS: `O(N*M)` N: rows, M; columns
- Space complexity: 
  - DFS: O(N*M)
  - BFS: O(min(N,M))
    - https://imgur.com/gallery/M58OKvB
    - Think about an example where dif(M, N) is big like 3x1000 grid. And the worst case is when we start from the middle of the grid.
Imagine how the processed points form a shape in the grid. It will be like a diamond and at some point, it will reach the longer edge of the grid. The possible shape at time t would be:
```sh
  ......QXXXQ.........
  .....QXXXXXQ........
  ......QXXXQ.........
```
So in this specific example (Q: points in the queue, .: not processed, X: processed) the number of the items in the queue is proportional with 3 because the smallest side limits the expanding.

## Feature #2: Copy Connections
- https://leetcode.com/problems/number-of-provinces/

**Problem**: As the user’s name can be the same, so every user on Facebook gets assigned a unique id. Each Facebook user’s connections are represented and stored in a graph-like structure. We will first have to make an exact copy of this structure before storing it on Instagram’s servers.

For each user, we’ll be provided with a node and its information. This node will point to other nodes, and from that one node, we need to reach and make an exact clone of every other node. An edge between two nodes means that they are friends with each other. A bi-directional edge means that both users follow each other, whereas a uni-directional edge means that only one user follows the other.

**Solution**: 
we use DFS traversal and create a copy of each node while traversing the graph. To avoid getting stuck in cycles, we’ll use a hashtable to store each completed node, and we will not revisit nodes that exist in the hashtable. The hashtable key will be a node in the original graph, and its value will be the corresponding node in the cloned graph.

**When we visit a node, we add it at the top of stack (DFS), add it to hashtable, create a new node (copy of that node).**

For the above graph, let’s assume the root (the randomly selected node from which we start the cloning process) is node 0. We’ll start with the root node 0.

```py
class Node:
    def __init__(self, val):
        self.val = val
        self.friends = []
        
# Main function
def clone(root):
    visited = {}
    return dfs(root, visited)


def dfs(root, visited):
    if root is None:
        return None
    
    new_node = Node(root.val)
    visited[root] = new_node

    for p in root.friends:
        x = visited.get(p)
        if x == None:
            new_node.friends += [dfs(p, visited)]
        else:
            new_node.friends += [x]
    return new_node
```

**Time complexity, space complexity: O(N)**

### Number of Provinces
- https://leetcode.com/problems/number-of-provinces/
**Solutions:**

**1. DFS**
The given matrix can be viewed as the Adjacency Matrix of a graph. By viewing the matrix in such a manner, our problem reduces to the problem of finding the number of connected components in an undirected graph. In this graph, the node numbers represent the indices in the matrix M and an edge exists between the nodes numbered ii and jj, if there is a 1 at the corresponding M[i][j].

In order to find the number of connected components in an undirected graph, one of the simplest methods is to make use of Depth First Search starting from every node. We make use of visitedvisited array of size N(M is of size NxN). This visited[i] element is used to indicate that the ith node has already been visited while undergoing a Depth First Search from some node.

To undergo DFS, we pick up a node and visit all its directly connected nodes. But, as soon as we visit any of those nodes, we recursively apply the same process to them as well. Thus, we try to go as deeper into the levels of the graph as possible starting from a current node first, leaving the other direct neighbour nodes to be visited later on.
```py
class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        visited = set()
        row = len(isConnected)
        count = 0
        
        for i in range(row):
            if i not in visited:
                self.dfs(isConnected, visited, i)
                count += 1
                    
        return count
    
    
    def dfs(self, grid, visited, i):
        row = len(grid)
        for j in range(row):
            if grid[i][j] == 1 and j not in visited:
                visited.add(j)
                self.dfs(grid, visited, j)
```
- Time complexity: `O(n^2)` the complete adjacency matrix of n^2 is traveresed.
- Space complexity: `O(n)` visited array

**2. BFS**:
```py
from collections import deque

class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        visited = set()
        row = len(isConnected)
        count = 0
        queue = deque()
        
        for i in range(row):
            if i not in visited:
                queue.append(i)
                while queue:
                    curr = queue.popleft()
                    visited.add(curr)
                    for j in range(row):
                        if isConnected[curr][j] == 1 and j not in visited:
                            queue.append(j)
                
                
                count += 1
                    
        return count
```

