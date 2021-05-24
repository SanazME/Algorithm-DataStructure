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

# DFS with stack:
- Given the root of a binary tree and an integer targetSum, return true if the tree has a root-to-leaf path such that adding up all the values along the path equals targetSum.

**DFS with Stack**
```py
# 1. DFS with stack
def hasPathSum(root, targetSum):
    if not root:
        return False
        
    stack = [(root, targetSum)]
    
    while stack:
        node, nodeSum = stack.pop()
        
        if node.val == nodeSum and not(node.left or node.right):
            return True
            
        hasPathSum(node.left, nodeSum - node.val)
        hasPathSum(node.right, nodeSum - node.val)
    return False
```
**DFS**
```py
# 1. DFS
def hasPathSum(root, targetSum):
    result = []
    dfs(root, targetSum, result)
    return any(result)
    
def dfs(node, nodeSum, result):
    if node:
        if node.val == nodeSum and not (node.left or node.right):
            result.append(True)
        if node.left: dfs(node.left, nodeSum - node.val, result)
        if node.right: dfs(node.right, nodeSum - node.val, result)
     
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

# # Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it is able to trap after raining.
# Example:
# Input: [0,1,0,2,1,0,1,3,2,1,2,1]
# Output: 6

- The brute force solution: for each location, find the max on the left and max on the right of that location and take the min of them
- Optimal solution: calculate the column of water in each location based on the followings:
    - If there was an infinite tall wall on the right end of the array, the water in each location would be the height of max so far on the left of the location - height of the location
    - Now for the locations on the right of the infinite wall, the water in each location coming from right to left is heigth of max so far on the right of the location - height of location.
