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
## 33. Compress File
- We have to come up with a compression strategy for text files that store our system’s information. Here is our strategy: whenever we see a word in a file composed as a concatenation of other smaller words in the same file, we will encode it with the smaller words’ IDs. For example, if we have the words `n, cat, cats, dog, and catsndog` in a text file. The word catsndog is the concatenation of the words n, cats, and dog, so we can assign it an ID. This way, instead of storing the entire `catsndog` word as is, we are storing an ID that takes much less space.

We’ll be provided a list of strings representing all the words from a text file. Our task will be to identify and isolate all the concatenated words.

- We’ll traverse the list of strings, and for each string, we’ll check every single combination. For example, some combinations of the word catsndog are `(c, atsndog), (ca, tsndog), (cat, sndog), (cats, ndog)`, etc. For each combination, we get two words. We can call the first word as prefix and the second word as suffix. Now, for a combination to be considered a concatenated word, both the prefix and suffix should be present in our list of strings. We’ll first perform the check for prefix because there is no need to check the suffix if the prefix is not present, and we can move to the next combination.

If the prefix word is present in our list of strings, then we move to check the suffix word. If the suffix word is also present, we have found the concatenated word. Otherwise, there can be two possibilities:

1. The suffix word is a concatenation of two or more words. We will recursively call the same function on the suffix word to generate more (prefix, suffix) pairs until a suffix that is present in our list of strings is found.

2. There is no such suffix word present in our list of strings, and our current combination is not a concatenated word.

When we break down the first suffix word, it breaks that word down to the last letter to check it in our list of strings. After this, we move to the next combination. This breakdown and search process occurs in DFS fashion which helps us form a tree-like structure for all words.

Now, for a single string, we generate multiple combinations, and for each of those combinations, we might recursively generate all consecutive sequences of the words. There will likely be an overlap in which we’ll compute the validity of a word multiple times, and this will increase our time complexity. For example, for the combination (c, atsndog), at some point, we will have to check the validity of the suffix word combination (a, tsndog). Now, when we get a combination (ca, tsndog) from a different iteration, we will again check the word tsndog when it has already been checked. The easiest way to counter this extra computation time is to use a cache. This way, if we encounter the same word again, instead of calling an expensive DFS function, we can simply return its value from the cache. This technique is also known as memoization.

Let’s see how we might implement this functionality:

1. Initialize the list of strings to a set data structure for O(1)O(1) lookups. Additionally, initialize a HashMap to be used as a cache.

2. Traverse the list of strings provided as input and call the DFS on each word.

3. In the DFS function, compute the prefix and suffix words.

4. If the prefix is found in our set, then search for suffix. If the suffix word is not found, then recursively call DFS on the suffix word.

5. If a word’s result is not calculated, we compute it during the above steps and cache (or memoize) it.

Otherwise, we get the result from the cache directly.

- **Time Complexity**: `O(N * M^2)`: Let n be the size of the list of strings and m be the average length of a string in the list of strings.
- There are O(m) different places where each word can be split into prefix and suffix. For every split, every character in the prefix and suffix are processed in the worst case, so that gives us O(m^2) for each word. Since there are n words, we will have O(n×m^2) time complexity.

- **Space Complexity**
- - In the worst case, all O(n) words will end up in res. This requires O(n \times m)O(n×m) space. The word_set requires another O(n \times m)O(n×m) space. Then, there is the cache. All n words become part of the cache, accounting for O(n \times m)O(n×m) space. In the worst case, all the suffixes and prefixes may be stored there as well. For one word, there are at most O(m)O(m) suffix and prefix pairs. The cumulative size of these prefixes and suffixes is O(m^2). **The first split results in the prefix of size 1, suffix m-1. Then, there is 2, m-2, and so on, resulting in O(m^2). This means the space complexity will be O(n * m^2)**

```py
def identify_concatenations(words):
    # Set for O(1) lookups
    word_set = set(words)
    
    def dfs(word, cache):
        # If result for current word already calculated then
        # return from cache
        if word in cache:
            return cache[word]
        # Traverse over the word to generate all combinations    
        for i in range(1, len(word)):
            # Divide the word into prefix and suffix
            prefix = word[:i]
            suffix = word[i:]
            
            if prefix in word_set:
                if suffix in word_set or dfs(suffix, cache):
                    cache[word] = True
                    return True
        cache[word] = False
        return False
    
    res = []
    cache = {}
    # Process for each word
    for word in words:
        if dfs(word, cache):
            res.append(word)
    return res

# Driver code

file_words = ["n", "cat", "cats", "dog", "catsndog"]
print("The following words will be compressed:", identify_concatenations(file_words))
```
