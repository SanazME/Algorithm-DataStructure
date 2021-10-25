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
- https://www.educative.io/courses/decode-coding-interview-python/NEOlg0Y5wND
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
## 34. Sequence construction
- https://leetcode.com/problems/sequence-reconstruction/
- Answer: https://leetcode.com/problems/sequence-reconstruction/discuss/92574/Very-short-solution-with-explanation

```py
def sequence_reconstruction(org, seqs):
  if len(seqs) == 0 or len(org) == 0:
    return False

  orgLen = len(org)
  idx = [0 for _ in range(orgLen+1)]
  pair = [False for _ in range(orgLen)]
  
  for i in range(orgLen):
    idx[org[i]] = i

  for seq in seqs:
    for i in range(len(seq)):
      if seq[i] > orgLen or seq[i] < 0:
        return False

      if i > 0 and idx[seq[i-1]] >= idx[seq[i]]:
        return False

      if i > 0 and idx[seq[i-1]] + 1 == idx[seq[i]]:
        pair[idx[seq[i-1]]] = True

  for i in range(len(pair) - 1):
    ele = pair[i]
    if not ele:
      return False

  return True


```
## 35. Identify Peak Interaction Times
- https://www.educative.io/courses/decode-coding-interview-python/mEKX514XyKp
- To solve this problem, we can start by calculating the sum of n intervals for each index, where n = hours. We can store these sums in an array called sums. Then, all that’s left will be to find the lexicographically smallest tuple of indices [a, b, c] that maximizes sums[a] + sums[b] + sums[c].

Let’s take a look at the algorithm to find the solution:

First, we need to calculate the sums at each index and store them in the array called sums. We can calculate sums using a sliding window.

When the sums are calculated, we need to find the [a, b, c] indices. We know that we can assume some constraints for the lexicographically smallest trio of indices.

If we consider b to be fixed, a should be in between 0 and b - n. Similarly, c should be between b + n and sums.length -1. We can deduce these constraints considering we want non-overlapping intervals. We can now find these indices using dynamic programming.

We will create two arrays, left and right. These arrays will store the maximum starting index from the left and right, respectively. The left[i] will contain the first occurrence of the largest value of W[a] on the interval a \in [0, i]a∈[0,i]. Similarly, the right[i] will be the same but on the interval a \in [i, \text{len}(sums) - 1]a∈[i,len(sums)−1].

Finally, for each value of b, we will check whether the corresponding left and right values are in the above-mentioned constraints or not. Out of all the trios that fulfill the constraints, we will choose the one that produces the maximum sums[a] + sums[b] + sums[c].

```py

def three_subarray_max_sum(numbers, k):
    sums = []
    curr_sum = 0
    # sum of k-element subarrays
    for i, ele in enumerate(numbers):
        curr_sum += ele

        if i >= k:
            curr_sum -= numbers[i - k]

        if i >= k - 1:
            sums.append(curr_sum)

    # Max sum indices from left
    left = [0 for _ in range(len(sums))]
    best = 0

    for i in range(len(sums)):
        if sums[i] > sums[best]:
            best = i
        left[i] = best

    # Max sum indices from right
    right = [0 for _ in range(len(sums))]
    best = len(sums) - 1

    for i in range(len(sums) - 1, -1, -1):
        if sums[i] > sums[best]:
            best = i
        right[i] = best

    re = []

    for mid in range(k, len(sums) - k):
        l = left[mid - k]
        r = right[mid + k]

        if len(re) == 0 or (sums[l] + sums[mid] + sums[r] > sums[re[0]] + sums[re[1]] + sums[re[2]]):
            re = [l, mid, r]

    return re
```

- My solution (not working with lexigraphically smallest one):
```py
from queue import PriorityQueue

def peak_interaction_times(interaction, hours):
    groups = 3
    result = []
    if len(interaction) < hours * groups:
        return []
    
    i = 0
    q = PriorityQueue()
    
    while i + hours <= len(interaction):
        sumTwo = sum(interaction[i:i+hours])
        q.put((-sumTwo, (i, i + hours - 1)))
        i += 1
      
    visited = set()
    while groups > 0:
        _, (start, end) = q.get()
        # print(ele)
        if start in visited or end in visited: continue
            
        visited.update([start, end])
        result.append(start)
        groups -= 1
    
    result.sort()
    return result
    
print(peak_interaction_times([0,2,1,3,1,7,11,5,5], 2))
```

- **To get a agg sum of elements over intervals of hours:**
```py
interactions = [0,2,1,3,1,7,11,5,5]
hours = 2
sums = []
curr_sum = 0

for i, ele in enumerate(interactions):
    curr_sum += x
    
    if i >= hours:
        curr_sum  -= interactions[i - hours]
    
    if i >= hours - 1:
        sums.append(curr_sum)
```

## Split Users into Two Groups
- https://www.educative.io/courses/decode-coding-interview-python/BngpZMyrVjY
- In this feature, the company has decided that they want to show people “follow” recommendations. For this purpose, we have been given the “following” relationship information for a group of users in the form of a graph. We want to see if these people can be split into two groups such that no one in the group follows or is followed-by anyone in the same group. We will then recommend people from the same group to each other.

The “following” relationship graph is given in the form of an undirected graph. So, if UserA follows UserB, or the other way around, it does not matter. This graph will be given to you as an input in the form of a 2D array called graph. In this array, each graph[i] will contain a list of indices, j, for which the edge between nodes i and j exists. Each node in the graph will represent a person and will be denoted by an integer ID from 0 to graph.length-1.

For example, the graph can be [[3], [2, 4], [1], [0, 4], [1, 3]].

- **Bigraph or Bipartite graph**
- To check if a graph is bipartite, we will color a node blue if it is part of the first set; otherwise, we will color it red. We can color the graph greedily if and only if it is bipartite. In a bipartite graph, all of a blue node’s neighbors must be red, and all of a red node’s neighbors must be blue.
- The complete algorithm is given below:

We’ll keep an array or HashMap to store each node’s color as color[node]. The possible values for colors can be 0, 1, or uncolored (-1 or null).

We will search each node in the graph to ensure disconnected nodes are also visited. For each uncolored node, we’ll start the coloring process by doing DFS on that node.

To perform DFS, we will first check if the nodes connected to the current node are colored or not. If a node is colored, we will check if the color is the same color as the current node. If the colors are the same, we return false.

If the node is not colored, we will color it and call DFS on that node recursively. If the recursive call returns false, the current DFS should also return false because coloring will not be possible.

If everything goes well and colors are assigned successfully, we will return true at the end.


```py
def is_split_possible(graph):
    color = {}
    def dfs(pos):
        for i in graph[pos]:
            if i in color:
                if color[i] == color[pos]:
                    return False
            else:
                color[i] = 1 - color[pos]
                if not dfs(i):
                    return False
        return True
    for i in range(len(graph)):
        if i not in color:
            color[i] = 0
            if not dfs(i):
                return False
    return True

# Driver code
graph = [[3], [2, 4], [1], [0, 4], [1, 3]]
print(is_split_possible(graph))
```
## 37. Add Binary numbers
- https://www.educative.io/courses/decode-coding-interview-python/R8kr2ZzZ2zK
```py
def add_binary(a, b):
    if not(a and b):
        return a or b
    
    carry = 0
    res = ''
    res = []
    
    na, nb = len(a), len(b)
    pa, pb = na - 1, nb - 1
    
    for i in range(max(na, nb) - 1, -1, -1):
        if pa >= 0:
            numa = ord(a[pa]) - ord('0')
            pa -= 1
        else:
            numa = 0
            
        if pb >= 0:
            numb = ord(b[pb]) - ord('0')
            pb -= 1
        else:
            numb = 0
            
        sumTwo = numa + numb + carry
        carry = sumTwo // 2
        res = str(sumTwo%2) + res
        res.append(sumTwo % 2)
        
    if carry == 1:
        res = str(carry) + res
        res.append(carry)
        
    return res
    return '.join([str(x) for x in res[::-1]
     
    
print(add_binary("10", "10111101"))
````
## 38. Design Search Autocomplete System
- https://www.educative.io/courses/decode-coding-interview-python/7npPkxyOD7r
- The second feature we want to implement is the auto-complete query. This is the feature that prompts the search engine to give you some suggestions to complete your query when you start typing something in the search bar. These suggestions are given based on common queries that users have searched already that match the prefix you have typed. Moreover, these suggestions are ranked based on how popular each query is.

Assume the search engine has the following history of queries: ["beautiful", "best quotes", "best friend", "best birthday wishes", "instagram", "internet"]. Additionally, you have access to the number of times each query was searched. The following list shows the number each query string occurred, respectively: [30, 14, 21, 10, 10, 15]. Given these parameters, we want to implement an autoComplete() function that takes in an incomplete query a user is typing and recommends the top three queries that match the prefix and are most popular.

The system should consider the inputs of the autoComplete() function as a continuous stream. For example, if the autoComplete("be") is called, and then the autoComplete("st") is called, the complete input at that moment will be "best". The input stream will end when a specific delimiter is passed. In our case, the delimiter is "#", and, autoComplete("#") will be called. This marks the end of the query string. At this point, your system should store this input string in the record, or if it already exists, it should increase its number of instances.

Suppose, the current user has typed "be" in the search bar; this will be the input for the autoComplete() function, meaning autoComplete("be") will be called. It will return ['beautiful', 'best friend', 'best quotes'] because these queries match the prefix and are the most popular. The order of queries in the output list is determined by popularity. Then, the user adds "st" to the query, making the string "best", and the autoComplete("st") will be called. Now, the output should be ['best friend', 'best quotes', 'best birthday wishes']. Lastly, the user finishes the query, so the autoComplete("#") will be called. It will output [] and "best" will be added in the record to be used in future suggestions.

- Solution:
- To design this system, we will again use the trie data structure. Instead of simply storing the words in the prefix tree, as we did in the WordDictionary, we will now store the query strings. The `AutocompleteSystem` class will act as a trie that keeps a record of the previous queries and assigns them a rank based on their number of occurrences.

We will implement the AutocompleteSystem class as follows:

- `Constructor`: In the constructor, we will feed the historical data into the system and create a trie out of it. We will initialize the root node of trie and call the `addRecord()` function to add all the records.

- `addRecord()` function: This function inserts records in the trie by creating new nodes. Its functionality is similar to the `insertWord()` function that we discussed in Feature #1: Store and Fetch Words. Each node of the trie will have:

    - A `children` dictionary
    - A Boolean called `isEnd` to set the end of the query sentence
    - A new variable called `data` that is optional, but we can use it **to store the whole query sentence in the last character of the sentence**. This will increase space complexity but can make computation easier.
    - A `rank` variable to store the number of occurrences
In the code below, you will notice that we are storing the rank as a negative value. There is a valid reason for doing this unintuitive step. Later, you will see that we will be using the `sorted()` method to obtain the top three results, and this negative rank will play a significant role. This will be explained in more detail in the explanation for the `autoComplete()` function.

- `search()` function: This function checks to see if the first character exists in its children beginning with the root node. If it exists, we move on to the node with the first character and check its children for the next character. If the node corresponding to a character is not found, we return []. If the search string is found, the dfs() helper function is called on the node with the last character of the input query.

- `dfs()` function: This function is the **helper function that returns all the possible queries in the record that match the input**. First, it will check if the node has isEnd set to True; if it does, the node’s rank and data will be appended as a tuple in an output list ret. Then, DFS exploration will be performed on all children nodes. All the possible queries will be added in ret and returned at the end.

- `autoComplete()` function: This function checks if the input string is not the end of string delimiter "#". If it is not the end of the input string, we append the value to the keyword member variable. Then, we call the `search()` function, which returns the list of tuples as discussed above. On the other hand, if the input string is the end of the input, we add the value present in keyword to the trie by calling `addRecord()`. Next, the value of the keyword variable is reset to an empty string. Before returning, you can see that we do some computation on the result list. The list that we received from the search() function was a list of tuples with rank and sentence as elements. We will sort the array in ascending order using the sorted() function and fetch the first three elements of the sorted list. From this list of three tuples, we will create a list containing only the second element of each tuple, meaning only the sentences. Finally, we return it. If we had used the actual positive value for rank, we would have needed to sort ascending for sentence and descending for rank. So, by making rank negative, we can easily sort the array in ascending using the default configuration in the sorted() function.

- My implementation with having a list of top 3 words for each node. It requires sorting N log N at each node:
```py
class TrieNode():
    def __init__(self):
        self.children = {}
        self.endOfWord = False
        self.suggestions = []
        self.rank = 0
        
class AutoCompleteSystem():
    def __init__(self, sentences, times):
        self.root = TrieNode()
        self.keyword = ''
        
        for i, sentence in enumerate(sentences):
            self.addRecord(sentence, times[i])
        
    def addRecord(self, sentence, hot):
        curr = self.root
        
        for char in sentence:
            if char not in curr.children:
                curr.children[char] = TrieNode()
            
            curr = curr.children[char]
            curr.suggestions.append((-hot, sentence))
            curr.suggestions.sort(key=lambda x: x[0])
            
            if len(curr.suggestions) > 3:
                curr.suggestions.pop()
        
        curr.endOfWord = True
      
    
    def search(self, word):
        curr = self.root
        result = []
        
        for char in word:
            if char not in word:
                return result
            
            curr = curr.children.get(char)
        
        result.extend(curr.suggestions if curr else [])
        return result
                
        
    def autoComplete(self, c):
        results = []
        if c != '#':
            self.keyword += c
            results = self.search(self.keyword)
        else:
            self.addRecord(self.keyword, 1)
            self.keyword = ""
        # print('result:', results)
        return [ele[1] for ele in results]
    
sentences = ["beautiful", "best quotes", "best friend", "best birthday wishes", "instagram", "internet"]
times = [30, 14, 21, 10, 10, 15]
auto = AutoCompleteSystem(sentences, times)
print(auto.autoComplete("b"))
print(auto.autoComplete("e"))
print(auto.autoComplete("s"))
print(auto.autoComplete("t"))
print(auto.autoComplete("#"))
   
```
- Other implementation that uses `dfs` helper function to get all possible words that match the input:
- **Time complexiy**:
    - `constructor`: The time complexity is `O(n×l)`, where n records of average length l are traversed to create the trie.
    - `autoComplete()`: The time complexity is `O(q + m + [l×log(l)])`, where q is the length of the query string at that moment and m is the number of nodes in the trie. The factor l×log(l) indicates the time taken to sort the list of queries to obtain the top three ranking queries.
- **Space complexity**:
    - constructor: The space complexity is `O(n×l)`, where nn records of average length ll are stored in the trie.
    - autoComplete(): The space complexity is `O(n×l)`, where nn records of average length ll are stored in the list to return at the end.
```py
class Node(object):
    def __init__(self):
        self.children = {}
        self.isEnd = False
        self.data = None
        self.rank = 0
        
class AutocompleteSystem(object):
    def __init__(self, sentences, times):
        self.root = Node()
        self.keyword = ""
        for i, sentence in enumerate(sentences):
            self.addRecord(sentence, times[i])

    def addRecord(self, sentence, hot):
        p = self.root
        for c in sentence:
            if c not in p.children:
                p.children[c] = Node()
            p = p.children[c]
        p.isEnd = True
        p.data = sentence
        p.rank -= hot
    
    def dfs(self, node):
        ret = []
        if node.isEnd:
            ret.append((node.rank, node.data))
        for child in node.children:
            ret.extend(self.dfs(node.children[child]))
        return ret
        
    def search(self, sentence):
        p = self.root
        for c in sentence:
            if c not in p.children:
                return []
            p = p.children[c]
        # print(self.dfs(p))
        return self.dfs(p)
    
    def autoComplete(self, c):
        results = []
        if c != "#":
            self.keyword += c
            results = self.search(self.keyword)
        else:
            self.addRecord(self.keyword, 1)
            self.keyword = ""
        return [item[1] for item in sorted(results)[:3]]

# Driver code
sentences = ["beautiful", "best quotes", "best friend", "best birthday wishes", "instagram", "internet"]
times = [30, 14, 21, 10, 10, 15]
auto = AutocompleteSystem(sentences, times)
print(auto.autoComplete("b"))
print(auto.autoComplete("e"))
print(auto.autoComplete("s"))
print(auto.autoComplete("t"))
print(auto.autoComplete("#"))
```
## 39. Word Break
- https://www.educative.io/courses/decode-coding-interview-python/JP2ZPmoLrKo
- You are given a non-empty string s and a list of strings called subs. The subs list will contain a unique set of strings. Your job is to determine if s can be broken down into a space-separated sequence of one or more strings from the subs list. A single string from subs can be reused multiple times in the breakdown of s.

Input#
The first input will be a non-empty string called s, and the second input will be a list string called subs. The following is an example input:

magically
["ag", "al", "icl", "mag", "magic", "ly", "lly"]

```py
def string_break(s, subs):
    # write your code here
                    
    subSet = set(subs)

    def dfs(s, cache):
        if s in cache:
            return cache[s]

        for i in range(len(s)):
            prefix = s[:i]
            suffix = s[i:]

            if prefix in subSet:
                if suffix in subSet or dfs(suffix, cache):
                    cache[s] = True
                    return True
        cache[s] = False
        return False


    cache = {}
    return dfs(s, cache)
```
## 40. Possible queries after adding white spaces
- https://www.educative.io/courses/decode-coding-interview-python/RLMDYggjEnR
- **Time complexity**: The time complexity is O(n^2 + 2^n + l) where n is the length of query and l is the length of the list containing words of the dictionary.

- **Space complexity**: The space complexity is O((n * 2^n) + l), where n is the length of the query and l is the length of the list containing the dictionary’s words.
```py
def break_query(query, dict):
    """
    :type query: str
    :type dict: List[str]
    :rtype: List[str]
    """
    return helper(query, set(dict), {})
    
def helper(query, dict, result):
    if not query: 
        return []
    
    if query in result: 
        return result[query]
    
    res = []
    for word in dict:
        if not query.startswith(word):
            continue
        if len(word) == len(query):
            res.append(word)
        else:
            resultOfTheRest = helper(query[len(word):], dict, result)
            for item in resultOfTheRest:
                item = word + ' ' + item
                res.append(item)
    result[query] = res
    return res

query = "vegancookbook"
dict = ["an", "book", "car", "cat", "cook", "cookbook", "crash", 
        "cream", "high", "highway", "i", "ice", "icecream", "low", 
        "scream", "veg", "vegan", "way"]
print(break_query(query, dict))
query = "highwaycarcrash"
print(break_query(query, dict))
```
## 41. Three Sums
- https://www.educative.io/courses/decode-coding-interview-python/R8q3n51LLjV
- https://leetcode.com/problems/3sum/

- **solution 1**:
    - it includes sorting the list and then ignoring the duplicate ones and also go till the element becomes positive, because if the ele1 is > 0 then in a sorted array all the other twos will be also postive and so we won't find a zero sum.

-**solution 2**: we don't need to sort the list. Let's start with sorted version and once the interviewer asked for non-sorting version, we can use this veraion.
- What if you cannot modify the input array, and you want to avoid copying it due to memory constraints?

- We can adapt the hashset approach above to work for an unsorted array. We can put a combination of three values into a hashset to avoid duplicates. Values in a combination should be ordered (e.g. ascending). Otherwise, we can have results with the same values in the different positions.

**Algorithm**

The algorithm is similar to the hashset approach above. We just need to add few optimizations so that it works efficiently for repeated values:

1. Use another hashset dups to skip duplicates in the outer loop.
    - Without this optimization, the submission will time out for the test case with 3,000 zeroes. This case is handled naturally when the array is sorted.

2. Instead of re-populating a hashset every time in the inner loop, we can use a hashmap and populate it once. Values in the hashmap will indicate whether we have encountered that element in the current iteration. When we process nums[j] in the inner loop, we set its hashmap value to i. This indicates that we can now use nums[j] as a complement for nums[i].

```py
def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res = []

        if len(nums) < 3:
            return []

        def twoSums(nums, i):
            seen = set()
            j = i + 1

            while j < len(nums):
                complement = -nums[i] - nums[j]

                if complement in seen:
                    res.append([nums[i], nums[j], complement])
                    while j+1 < len(nums) and nums[j] == nums[j+1]:
                        j += 1

                seen.add(nums[j])
                j += 1

        for i, num1 in enumerate(nums):
            if num1 > 0: 
                break
            if i == 0 or nums[i-1] != nums[i]:
                twoSums(nums, i)

        return res
```
```py
def threeSum(nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    if len(nums) < 3:
        return []
    
    dup = set()
    res = set()
    seen = {}
    
    for i, num1 in enumerate(nums):
        if num1 not in dup:
            dup.add(num1)
            for j, num2 in enumerate(nums[i+1:]):
                complement = - num1 - num2
                if complement in seen and seen[complement] == i:
                    res.add(tuple(sorted((num1, num2, complement))))
                seen[num2] = i
    return res
        
    
    
print(threeSum([0,0,0]))
print(threeSum([-1,0,1,2,-1,-4]))
```
## 42. Copy List with Random Pointer
- https://leetcode.com/problems/copy-list-with-random-pointer/
- To make a deep copy of the list, we will iterate over the original list and create new nodes via the related pointer or the next pointer. We can also use a Hashtable/dictionary to track whether the copy of a particular node is already present or not.

The complete algorithm is as follows:

We will traverse the linked list starting at the `head`.

We will use a dictionary/Hashtable to keep track of `visited` nodes.

We will make a copy of the current node and store the old node as the key and the new node as the value in `visited`.

If the `related` pointer of the current node, ii, points to the node jj and a clone of jj already exists in `visited`, we will use the cloned node as a reference.

Otherwise, if the `related` pointer of the current node, ii, points to the node jj, which has not been created yet, we will create a new node that corresponds to jj and add it to `visited`.

If the `next` pointer of the current node, ii, points to the node jj and a clone of jj already exists in visited, we will use the cloned node as a reference.

Else, if the `next` pointer of the current node, ii, points to the node jj, which has not been created yet, we will create a new node corresponding to jj and add it to the visited dictionary.

```py
"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""

class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        if head == None:
            return None
        
        def getClonedNode(node):
            if node:
                if node in visited:
                    return visited[node]
                else:
                    visited[node] = Node(node.val, None, None)
                    return visited[node]
                
            return None
        
        visited = {}
        curr = head
        newNode = Node(curr.val, None, None)
        visited[curr] = newNode
        
        while curr:
            newNode.next = getClonedNode(curr.next)
            newNode.random = getClonedNode(curr.random)
            
            newNode = newNode.next
            curr = curr.next
            
        return visited[head]
```
- another solution which give `O(1)` space complexity so we don't have to construct a hashTable to keep track of mapping betweent the old and new nodes is to intervene new nodes within old nodes.
- Instead of a separate dictionary to keep the old node --> new node mapping, we can tweak the original linked list and keep every cloned node next to its original node. This interleaving of old and new nodes allows us to solve this problem without any extra space. Lets look at how the algorithm works.
    1. Traverse the original list and clone the nodes as you go and place the cloned copy next to its original node. This new linked list is essentially a interweaving of original and cloned nodes.
    2. As you can see we just use the value of original node to create the cloned copy. The next pointer is used to create the weaving. Note that this operation ends up modifying the original linked list.
    3. Iterate the list having both the new and old nodes intertwined with each other and use the original nodes' random pointers to assign references to random pointers for cloned nodes. For eg. If B has a random pointer to A, this means B' has a random pointer to A'.
    4. Now that the random pointers are assigned to the correct node, the next pointers need to be correctly assigned to unweave the current linked list and get back the original list and the cloned list.
```py
class Solution(object):
    def copyRandomList(self, head):
        """
        :type head: Node
        :rtype: Node
        """
        if not head:
            return head

        # Creating a new weaved list of original and copied nodes.
        ptr = head
        while ptr:

            # Cloned node
            new_node = Node(ptr.val, None, None)

            # Inserting the cloned node just next to the original node.
            # If A->B->C is the original linked list,
            # Linked list after weaving cloned nodes would be A->A'->B->B'->C->C'
            new_node.next = ptr.next
            ptr.next = new_node
            ptr = new_node.next

        ptr = head

        # Now link the random pointers of the new nodes created.
        # Iterate the newly created list and use the original nodes random pointers,
        # to assign references to random pointers for cloned nodes.
        while ptr:
            ptr.next.random = ptr.random.next if ptr.random else None
            ptr = ptr.next.next

        # Unweave the linked list to get back the original linked list and the cloned list.
        # i.e. A->A'->B->B'->C->C' would be broken to A->B->C and A'->B'->C'
        ptr_old_list = head # A->B->C
        ptr_new_list = head.next # A'->B'->C'
        head_new = head.next
        while ptr_old_list:
            ptr_old_list.next = ptr_old_list.next.next
            ptr_new_list.next = ptr_new_list.next.next if ptr_new_list.next else None
            ptr_old_list = ptr_old_list.next
            ptr_new_list = ptr_new_list.next
        return head_new
```
## 43. Find the fist and last occurance of a number in a sorted array
- we use binary search twice to find the start and end of that occurance. For the end, we're coming from left and so first update the `left` var. For start, we're coming from right and so update `right` first.
- **Time Complexity**: `O(log n)`

```py
def findRange(numbers, target):
    start = -1
    end = -1

    if len(numbers) == 0:
        return [start, end]

    start = findStart(numbers, target)
    end = findEnd(numbers, target)

    return [start, end]


def findStart(numbers, target):
    left, right = 0, len(numbers) - 1
    
    while left < right:
        mid = (left + right) // 2

        if numbers[mid] >= target:
            right = mid
        else:
            left = mid + 1

    if numbers[left] == target:
        return left
    else:
        return -1


def findEnd(numbers, target):
    left, right = 0, len(numbers) - 1

    while left + 1 < right:
        mid = (left + right) // 2

        if target >= numbers[mid]:
            left = mid
        else:
            right = mid
    
    if numbers[right] == target:
        return right
    elif numbers[left] == target:
        return left
    else:
        return -1
```
## 44. Products Frequently Viewed Together
- https://www.educative.io/courses/decode-coding-interview-python/YM946D429oA
- also similar to find all anagrams in here: https://github.com/SanazME/Algorithm-DataStructure/blob/master/AmazonQuestions.md#21-find-all-anagrams-in-a-string

```py
import collections

def findSimilarity(products, candidates):
    if len(products) < len(candidates):
        return []
    
    candidateCount = collections.Counter(candidates)
    productCount = collections.Counter()
    
    output = []
    
    for i, num in enumerate(products):
        
        productCount[num] += 1
        
        if i >= len(candidates):
            # remove the first element from set
            productCount[products[i-len(candidates)]] -= 1
            
            if productCount[products[i-len(candidates)]] == 0:
                del productCount[products[i-len(candidates)]]
        
        if i >= len(candidates) - 1:
            # compare two sets or lists
            if productCount == candidateCount:
                output.append(i - len(candidates) + 1)
                
    return output


print(findSimilarity([3, 2, 1, 5, 2, 1, 2, 1, 3, 4], [1,2,3]))                
```
## 45. Maximum Units on a Truck
- https://leetcode.com/problems/maximum-units-on-a-truck/

```py
from queue import PriorityQueue
class Solution:
    def maximumUnits(self, boxTypes: List[List[int]], truckSize: int) -> int:
        q = PriorityQueue()
        
        for item in boxTypes:
            count, weight = item[0], item[1]
            q.put((-weight, count))
            
        remaining = truckSize
        totalWeight = 0
        
        while not q.empty():
            weight, count = q.get()
            soFar = min(remaining, count)
            totalWeight += soFar * -weight
            remaining -= soFar
            
            if remaining == 0:
                return totalWeight
            
        return totalWeight

```
## 46. Merge Recommendations
- https://www.educative.io/courses/decode-coding-interview-python/B10JG6jJQkX
- Accounts Merge
- This feature can be mapped to a graph problem. We draw an edge between two emails, in case they occur in the same account. From here, the problem comes down to finding the connected components of this graph.

The complete algorithm is as follows:

    - First, we will build an undirected graph by creating an edge from the first email to all the other emails in the same account. Each email is treated as a node and an adjacency graph will be made.
    - Additionally, we’ll remember a map from emails to names on the side.
    - Now, we will use a depth-first search starting with the first email.
    - We will find all the nodes/emails that can be reached from the current email and denote it as a connected component. Then, we will add the respective name and this component, in sorted order, to the final answer.
    - We will keep track of the visited nodes. If a visited node is found, this means that it was already a part of a previous component, so we can skip it.

```py
import collections
def accountsMerge(accounts):
    email_to_name = {}
    graph = collections.defaultdict(set)
    
    for account in accounts:
        name = account[0]
        emails = account[1:]
        email1 = emails[0]
        email_to_name[email1] = name
        
        for email in emails[1:]:
            graph[email1].add(email)
            graph[email].add(email1)
            email_to_name[email] = name
            
    # DFS
    visited = set()
    output = []
    
    for email in graph:
        if email not in visited:
            visited.add(email)
            component = []
            stack = [email]
            
            while stack:
                node = stack.pop()
                component.append(node)
                
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            
            output.append([email_to_name[email]] + component)
            
    return output
            
# Driver code
accounts = [["Sarah", "sarah22@email.com", "sarah@gmail.com", "sarahhoward@email.com"],
            ["Alice", "alicexoxo@email.com", "alicia@email.com", "alicelee@gmail.com"],
            ["Sarah", "sarah@gmail.com", "sarah10101@gmail.com"],
            ["Sarah", "sarah10101@gmail.com", "misshoward@gmail.com"]]
print(accountsMerge(accounts))
```
## 47. Remove negative nodes and their children from a tree
- Given a tree, remove all nodes and their children that have negative value.

```py
class Node():
    def __init__(self, val=0):
        self.val = val
        self.children = set()
        
def remove_negative_nodes(root):
    if not root:
        return None
        
    return helper(root, None)
 
def helper(node, parent):
    if not node:
        return
        
    stack = [(node, parent)]
    
    while stack:
        curr, currParent = stack.pop()
        if curr.val < 0:
            if currParent:
                currParent.children.remove(curr)
            else:
                return None
        else:
            for child in curr.children:
                stack.append((child, curr))
                
    return node
```
    
## 48. Products in Price Range
- https://www.educative.io/courses/decode-coding-interview-python/YQWrxNW5Gy2
- The product data is given to us in the form of a binary search tree, where the values are the prices. You will be given the parameters low and high; these represent the price range the user selected. This range is inclusive. Return all the data between low and high range.
- **How to create a Binary Search Tree**
```py
class Node:
    def __init__(self, val):
        self.val = val
        self.leftChild = None
        self.rightChild = Nonde
        
    def insert(self, val):
        if self is None:
            self = Node(val)
            return
        current = self
        
        while current:
            parent = current
            if val < current.val:
                current = current.leftChild
            else:
                current = current.rightChild
                
        if val < parent.val:
            parent.leftChild = Node(val)
        else:
            parent.rightChild = Node(val)
                
```
```py
class BinarySearchTree:
    def __init__(self):
        self.root = None
        
    def insert(self, val):
        if self.root is None:
            self.root = Node(val)
        else:
            self.root.insert(val)

```
```py
def producyInRange(root, low, high):
    products = []
    
    def preorder(node):
        if node:
            if low <= node.val <= high:
                output.append(node.val)
            if low <= node.val:
                preorder(node.leftChild)
            if node.val <= high:
                preorder(node.rightChild)
    
    
    preorder(root)
    return output

```

## 49 Subarray Product Less than K
- https://leetcode.com/problems/subarray-product-less-than-k/

## 50 Find Pivot Index
- https://leetcode.com/problems/find-pivot-index/

## 51. Group anagrams
- First, we need to figure out a way to individually group all the character combinations of each title. Suppose the content library contains the following titles: "duel", "dule", "speed", "spede", "deul", "cars". How would you efficiently implement a functionality so that if a user misspells speed as spede, they are shown the correct title?

We want to split the list of titles into sets of words so that all words in a set are anagrams. In the above list, there are three sets: {"duel", "dule", "deul"}, {"speed", "spede"}, and {"cars"}. Search results should comprise all members of the set that the search string is found in. We should pre-compute these sets instead of forming them when the user searches a title.

- **Solution**:
1. For each title, compute a 26-element vector. Each element in this vector represents the frequency of an English letter in the corresponding title. This frequency count will be represented as a tuple. For example, abbccc will be represented as (1, 2, 3, 0, 0, ..., 0). This mapping will generate identical vectors for strings that are anagrams.

2. Use this vector as a key to insert the titles into a Hash Map. All anagrams will be mapped to the same entry in this Hash Map. When a user searches a word, compute the 26-element English letter frequency vector based on the word. Search in the Hash Map using this vector and return all the map entries.

3. Store the list of the calculated character counts in the same Hash Map as a key and assign the respective set of anagrams as its value.

4. Return the values of the Hash Map, since each value will be an individual set.

`{ (2, 1, 1, 0, ..., 0) : [abac, aabc, baca, caab]}`

**Time complexity**:
- `O(n X k)` where `n` is the size of the list of strings and `k` is the maximum lenght that a single string can have.
**Space complexity**:`O(n X k)`

```py
def group_titles(titles):
    result = []
    if not titles:
        return result
    
    hashTable = {}
    
    for title in titles:
        vec = [0 for _ in range(26)]
        
        for char in title:
            idx = ord(char) - ord('a')
            vec[idx] += 1
        
        key = tuple(vec)
        if key in hashTable:
            hashTable[key].append(title)
        else:
            hashTable[key] = [title]
        
    print(hashTable)
    
titles = ["duel","dule","speed","spede","deul","cars"]  
print(group_titles(titles))
```
