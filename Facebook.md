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
- https://leetcode.com/problems/clone-graph/

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
OR define a separate dfs function:

```py
"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""

class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        visited = {}
        
        if not node:
            return node
        
        return self.dfs(node, visited)
    
    
    def dfs(self, node, visited):
        if not node:
            return node
        
        if node in visited:
            return visited[node]
        
        clone = Node(node.val, [])
        visited[node] = clone
        
        if node.neighbors:
            clone.neighbors = [self.dfs(n, visited) for n in node.neighbors]
            
        return clone
```

**Time complexity, space complexity: O(N)**

### 2.1 Number of Provinces
- https://leetcode.com/problems/number-of-provinces/
**Solutions:**

**Note**: this is slightly different than number of islands problem because the grid here is always N*N (as opposed to N*M) and also the diagoanl line is always 1 because each city is connected to itself and also is it mirrored across the diagonal line because if a is connected to b then b is connected to a (undirectional graph).

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
                ## can move to bfs function:
                queue.append(i)
                while queue:
                    curr = queue.popleft()
                    if curr in visited: continue
                    visited.add(curr)
                    for j in range(row):
                        if isConnected[curr][j] == 1 and j not in visited:
                            queue.append(j)
                ######
                
                count += 1
                    
        return count
```

### 2.2 Clone Graph
- https://leetcode.com/problems/clone-graph/

**Solution**:
The basic intuition for this problem is to just copy as we go. We need to understand that we are dealing with a graph and this means a node could have any number of neighbors. This is why neighbors is a list. What is also crucial to understand is that we don't want to get stuck in a cycle while we are traversing the graph. According to the problem statement, any given undirected edge could be represented as two directional edges. So, if there is an undirected edge between node A and node B, the graph representation for it would have a directed edge from A to B and another from B to A. After all, an undirected graph is a set of nodes that are connected together, where all the edges are bidirectional.

To avoid getting stuck in a loop we would need some way to keep track of the nodes which have already been copied. By doing this we don't end up traversing them again

**DFS Algorithm**
1. Start traversing the graph from the given node.
2. We would take a hash map to store the reference of the copy of all the nodes that have already been visited and cloned. The key for the hash map would be the node of the original graph and corresponding value would be the corresponding cloned node of the cloned graph. If the node already exists in the `visited` we return corresponding stored reference of the cloned node.
3. If we don't find the node in the visited hash map, we create a copy of it and put it in the hash map. Note, how it's important to create a copy of the node and add to the hash map before entering recursion.
```py
   clone_node = Node(node.val, [])
   visited[node] = clone_node
```

In the absence of such an ordering, we would be caught in the recursion because on encountering the node again in somewhere down the recursion again, we will be traversing it again thus getting into cycles.

4. Now make the recursive call for the neighbors of the node. Pay attention to how many recursion calls we will be making for any given node. **For a given node the number of recursive calls would be equal to the number of its neighbors**. Each recursive call made would return the clone of a neighbor. We will prepare the list of these clones returned and put into neighbors of clone `node` which we had created earlier. This way we will have cloned the given `node` and it's `neighbors`.

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

- Time Complexity: O(N + M), where N is a number of nodes (vertices) and M is a number of edges.
- Space Complexity: O(N). This space is occupied by the visited hash map and in addition to that, space would also be occupied by the recursion stack since we are adopting a recursive approach here. The space occupied by the recursion stack would be equal to O(H) where H is the height of the graph. Overall, the space complexity would be O(N).

**BFS Algorithm**
**Instead of stack in DFS, we use queue in BFS**
- We could agree DFS is a good enough solution for this problem. However, if the recursion stack is what we are worried about then DFS is not our best bet. Sure, we can write an iterative version of depth first search by using our own stack. However, we also have the BFS way of doing iterative traversal of the graph and we'll be exploring that solution as well.

The difference is only in the traversal of DFS and BFS. As the name says it all, DFS explores the depths of the graph first and BFS explores the breadth. Based on the kind of graph we are expecting we can chose one over the other. We would need the visited hash map in both the approaches to avoid cycles.

**Algorithm**
1. We will use a hash map to store the reference of the copy of all the nodes that have already been visited and copied. The key for the hash map would be the node of the original graph and corresponding value would be the corresponding cloned node of the cloned graph. The visited is used to prevent cycles and get the cloned copy of a node.
2. Add the first node to the queue. Clone the first node and add it to visited hash map.
3. Do the BFS traversal:
  - pop a node from the fron of the queue
  - visit all the neightbors of this node
  - if any of the neighbors was already visited then it must be present in the visited dictionary. Get the clone of this neighbor form visited in that case.
  - Otherwise, create a clone and store in the visited.
  - Add the clones of the neighbors to the corresponding list of the clone node.

```py
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val, neighbors):
        self.val = val
        self.neighbors = neighbors
"""
from collections import deque
class Solution(object):

    def cloneGraph(self, node):
        """
        :type node: Node
        :rtype: Node
        """

        if not node:
            return node

        # Dictionary to save the visited node and it's respective clone
        # as key and value respectively. This helps to avoid cycles.
        visited = {}

        # Put the first node in the queue
        queue = deque([node])
        # Clone the node and put it in the visited dictionary.
        visited[node] = Node(node.val, [])

        # Start BFS traversal
        while queue:
            # Pop a node say "n" from the from the front of the queue.
            n = queue.popleft()
            # Iterate through all the neighbors of the node
            for neighbor in n.neighbors:
                if neighbor not in visited:
                    # Clone the neighbor and put in the visited, if not present already
                    visited[neighbor] = Node(neighbor.val, [])
                    # Add the newly encountered node to the queue.
                    queue.append(neighbor)
                # Add the clone of the neighbor to the neighbors of the clone node "n".
                visited[n].neighbors.append(visited[neighbor])

        # Return the clone of the node from visited.
        return visited[node]
```
- Time Complexity: O(N + M), where N is a number of nodes (vertices) and M is a number of edges.
- Space Complexity: O(N). This space is occupied by the visited hash map and in addition to that, space would also be occupied by the queue since we are adopting the BFS approach here. The space occupied by the queue would be equal to O(W) where W is the width of the graph. Overall, the space complexity would be O(N).

## Feature #3: Find Story ID
- https://leetcode.com/problems/search-in-rotated-sorted-array/
We’ll have an array containing the story id’s. Some of the stories will be watched and will be at the end of the array, and some will be unwatched and will be at the start of the array. If a user clicks a story, we need to search for its id in the array and return the index of that id.

- Binary search
- **Time complexity**: `O(log N)`
- **Space complexity**: `O(1)`

```py
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        start, end = 0, len(nums) - 1
        
        while start <= end:
            mid = (start + end) // 2
            
            if nums[mid] == target:
                return mid
            
            elif nums[mid] >= nums[start]:
                if target >= nums[start] and target < nums[mid]:
                    end = mid - 1
                else:
                    start = mid + 1
            else:
                if target > nums[mid] and target <= nums[end]:
                    start = mid + 1
                else:
                    end = mid -1
                    
        return -1
```
## Feature #5: Flag Words
- https://leetcode.com/problems/expressive-words/

**Solution**
Since we have to observe letters of two strings at a time, we can follow a two-pointer approach to solve this problem.
Here is how we will implement this feature:
1. Initialize two pointers, i and j, to start traversing from S and W, respectively.
2. Check if letters currently pointed to by i and j of both words are equal. Otherwise, return False.
3.For each equal letter found:
4.Get the length of the repeating sequences of the equal letter found in both words.
5.The length of the repeating sequence of W letters should be less than or equal to the length of the repeating sequence of S letters.
6.The length of the repeating sequence of S letters should be greater than or equal to 3.
7.If any of the conditions mentioned in step 3 fails, return False.
8.If the ends of both strings are reached, return True.

```py
def flag_words(s, w):
    
    if not s or not w:
        return False

    i, j = 0, 0
    while i < len(s) and j < len(w):
        if s[i] != w[j]:
            return False
        else:
            len1 = repeated_letter(s,i)
            len2 = repeated_letter(w,j)
            
            print("char: ", s[i])
            print("len1: ", len1, "len2: ", len2)
            
            if (len1 >= 3 and len1 < len2) or (len1 < 3 and len1 != len2):
                return False
            i += len1
            j += len2
            
    return (i == len(s)) and (j == len(w))

def repeated_letter(s, i):
        tmp = i
        while tmp < len(s) and s[tmp] == s[i]:
            tmp += 1

        return tmp - i

S = "mooooronnnn" # modified word
W = "moron" # original word

if flag_words(S, W):
    print("Word Flagged")
    print("The word", '"' + S + '"', "is a possible morph of", '"' + W + '"')
else:
    print("Word Safe")
```

- Time complexity: `O(max(n,m))`
- Space complexity: `O(1)`

```py
class Solution:
    def expressiveWords(self, s: str, words: List[str]) -> int:
        if not s or not words:
            return False
        
        def repeated_count(s, i):
            tmp = i
            while tmp < len(s) and s[tmp] == s[i]:
                tmp += 1
            
            return tmp - i
        
        
        count = 0
        for w in words:
            i, j = 0, 0
            
            while i < len(s) and j < len(w):
                if s[i] == w[j]:
                    
                    len1 = repeated_count(s, i)
                    len2 = repeated_count(w, j)
                    
                    if (len1 >= 3 and len1 < len2) or (len1 < 3 and len1 != len2):
                        break
                    else:
                        i += len1
                        j += len2
                    
                else:
                    break
                    
            if i == len(s) and j == len(w):
                count += 1
                    
        return count
```

## Feature #6: Combine Similar Messages
- https://leetcode.com/problems/group-shifted-strings/

**Solution**:
From the above example, we can see that the difference between consecutive characters of each word is equal for each set. Consider the words lmn and mno from Set 1 of the example above. The difference between the ASCII values of each pair of consecutive characters of lmn is (1, 1), respectively, and the difference between each character of mno is also (1, 1). Words that are shifted versions of each other have identical character code differences. Using this, we can combine shifted words into separate sets. We can use a HashMap in which the keys can be represented by the differences between adjacent characters. The words that have these differences between their characters will be mapped as the values of these keys. For example, (1,1) will be a key with lmn and mno as its values. When all words are processed, the values of our HashMap will be the different groups.

In Set 2 of the above example, a wrap-around case occurs. In the string azb and bac, z(122) - a(97) gives us 25, but a(97) - b(98) gives us -1. If we don’t take care of this case, these two messages would end up being in two different sets. So, if any difference is less than zero, simply add 26 to it. For our case, -1 + 26 gives us 25, which is correct since, in reality, a is 25 steps away from b moving in a forward direction.

**ord() returns an integer representing the Unicode character.**

```py
class Solution:
    def groupStrings(self, strings: List[str]) -> List[List[str]]:
        
        def diff(string):
            rr = []
            for i in range(1, len(string)):
                offset = ord(string[i]) - ord(string[i-1])
                if offset < 0:
                    offset += 26
                rr.append(offset)
            re = tuple(rr)
                
            return re
                
        
        hashMap = {}
        
        for string in strings:
            key = diff(string)
            
            if key in hashMap:
                hashMap[key].append(string)
            else:
                hashMap[key] = [string]
                
        output = []
        
        print(hashMap)
        for k in hashMap.keys():
            output.append(hashMap[k])
            
        return output
```
- Time complexity: `O(N*M)` : N the length of strings and K the max lenght of a string.
- Space complexity: `O(N*M)` : We need to store all the strings plus their Hash values in hashMap.
