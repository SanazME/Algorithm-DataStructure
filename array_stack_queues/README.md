# Breadth First Search (BFS)

- One common application of Breadth-first Search (BFS) is to find the shortest path from the root node to the target node. BFS of a graph is similar to BFS of a tree. The only catch is, unlike tree, graphs may contain cycles. so we may come to th same node.To avoid processing a node more than once, we use a boolean visited array. 
- https://www.programiz.com/dsa/graph-bfs
- If a node X is added to the queue in the kth round, the length of the shortest path between the root node and X is exactly k. That is to say, you are already in the shortest path the first time you find the target node.

```py
# BFS algorithm in Python
import collections

def bfs(graph, root):
    queue = collections.deque([root])
    visited = set()
    visited.add(root)
    
    while queue:
        cur = queue.popleft()
        print(cur)
        
        for neighbor in graph[cur]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
            
if __name__ == '__main__':
    graph = {0: [1, 2], 1: [2], 2: [3], 3: [1, 2]}
    print("Following is Breadth First Traversal: ")
    bfs(graph, 0)
```
