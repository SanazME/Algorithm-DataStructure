## 30. Find all contiguous subarrays that sum up to the given n value
- https://www.educative.io/courses/decode-coding-interview-python/xopDqXlkGrq
- We can create cumulative sum at each indices and have a hashTable to save the cum sum as a key and its occurance as value. Then if the difference between two cumSum at indices `cumSum[i] - cumSum[j] = n` it means that the sum of numbers in between these two indices is n.

```py
import collections
def allocateSpace(processes, n):
    cumSum = collections.defaultdict(int)
    cumSum[0] = 1
    count = 0
    totalSum = 0
    
    for process in processes:
        totalSum += process
        
        print('totalSum - n: ', totalSum - n)
        if totalSum - n in cumSum:
            count += cumSum[totalSum - n]
            
        cumSum[totalSum] += 1
        print(cumSum)
        
    return count
```
## 31. Resume nth preempted process in the nth round of process resumption
- https://www.educative.io/courses/decode-coding-interview-python/RLP064XLg8R
- We're finding a nth missing number is a sorted array of processes IDs. We use binary search since it's sorted array.
- recursice and iterative solutions: 
```py
def resumeProcess(arr, n):
    
    left, right = 0, len(arr) - 1
    
    while left + 1 < right:
        mid = (left + right) // 2
        missing = arr[mid] - arr[left] - (mid - left)
        
        if n > missing:
            # on the right half
            left = mid
            n -= missing
        else:
            right = mid

        
    return arr[left] + n
    
# recursive
def resumeProcess(arr, n):
    
    def helper(left, right, n):
        # base case
        if left + 1 == right:
            return arr[left] + n
        
        mid = (left + right) // 2
        missing = arr[mid] - arr[left] - (mid - left)
        if n > missing:
            # right half
            return helper(mid, right, n - missing)
        else:
            return helper(left, mid, n)
    
    
    pid = helper(0, len(arr) - 1, n)
    
    return pid

```
## 32. Schedule processes
- https://www.educative.io/courses/decode-coding-interview-python/gx3N0GX2NkY
- When a system is booted, the operating system needs to run a number of processes. Some processes have dependencies, which are specified using ordered pairs of the form (a, b); this means that process b must be run before process a. Some processes don’t have any dependencies, meaning they don’t have to wait for any processes to finish. Additionally, there cannot be any circular dependencies between the processes like (a, b)(b, a). In order to successfully start the system, the operating system needs to select an ordering to run the processes. The processes should be ordered in such a way that whenever a process is scheduled all of its dependencies are already met.

We’ll be provided with the total number of processes, n, and a list of process dependencies. Our task is to determine the order in which the processes should run. The processes in the dependency list will be represented by their ID’s.

- **Solution: **
- The vertices in the graph represent the processes, and the directed edge represents the dependency relationship. From the above example, we get the order: P6 ➔ P4 ➔ P1 ➔ P5 ➔ P2 ➔ P3. Another possible ordering of the above processes can be P6 ➔ P4 ➔ P5 ➔ P1 ➔ P2 ➔ P3. This order of graph vertices is known as a Topological Sorted Order.

The basic idea behind the topological sort is to provide a partial ordering of the graph’s vertices such that if there is an edge from U to V; then U ≤ V; this means U comes before V in the ordering. Here are a few of the fundamental concepts of topological sorting:

Source: Any vertex that has no incoming edge and has only outgoing edges is called a source.

Sink: Any vertex that has only incoming edges and no outgoing edge is called a sink.

So, we can say that a topological ordering starts with one of the sources and ends at one of the sinks.

A topological ordering is possible only when the graph has no directed cycles, i.e., if the graph is a Directed Acyclic Graph (DAG). If the graph has one or more cycles, no linear ordering among the vertices is possible.

- To find the topological sort of a graph, we can traverse the graph in a Breadth First Search (BFS) manner. We will start with all the sources, and in a stepwise fashion, save all of the sources to a sorted list. We will then remove all of the sources and their edges from the graph. After removing the edges, we will have new sources, so we will repeat the above process until all vertices are visited.

Here is how we will implement this feature:

**1. Initialization**

- We will store the graph in adjacency lists, in which each parent vertex will have a list containing all of its children. We will do this using a HashMap, where the key will be the parent vertex number and the value will be a list containing the children vertices.

- To find the sources, we will keep a HashMap to count the in-degrees, which is the count of incoming vertices’ edges. Any vertex with a 0 in-degree will be a source.

**2. Build the graph and find in-degrees of all vertices**

- We will build the graph from the input and populate the in-degrees HashMap.

**3. Find all sources**

- Our sources will be all the vertices with 0 in-degrees, and we will store them in a queue.

**4.Sort**

- For each source, we’ll do the following:

        - Add it to the sorted list.

        - Retrieve all of its children from the graph.

        - Decrement the in-degree of each child by 1.

        - If a child’s in-degree becomes 0, add it to the source queue.

Repeat step 1, until the source queue is empty.

```py
import collections
def schedule_process(n, edges):
    
    # initialize parent hash and in-degree hash
    parentHash = {i: [] for i in range(n)} 
    in_degree = {i: 0 for i in range(n)}
    
    for ele in edges:
        child, parent = ele[0], ele[1]
        parentHash[parent].append(child)
         
        if parent not in in_degree:
            in_degree[parent] = 0
        
        in_degree[child] += 1
        
    print(parentHash, in_degree)
    
    # find sources/roots
    queue = collections.deque()
    for key in in_degree:
        if in_degree[key] == 0:
            queue.append(key)
    if len(queue) == 0:
        return []
    
    # DFS
    visited = set()
    count = 0
    
    sorted_list = []
    while queue:
        node = queue.popleft()
        sorted_list.append(node)
        
        for child in parentHash[node]:
            in_degree[child] -= 1
            
            if in_degree[child] == 0:
                queue.append(child)
                
    return sorted_list
        
processes = 7
arr = [[4,6],[5,6], [1,4], [2,1], [3,2], [5,4], [5,0], [3,5], [3,0]]
```
