## Articles and exercises


- **Tech Dose (youtube):** https://www.youtube.com/c/TECHDOSE4u/search?query=dynamic%20programming
- **strating point:** https://leetcode.com/problems/min-cost-climbing-stairs/solution/
- **Dynamic Programming patterns:** https://leetcode.com/discuss/general-discussion/458695/Dynamic-Programming-Patterns
- **ABCs of Greedy:** https://leetcode.com/discuss/general-discussion/1061059/ABCs-of-Greedy
- https://leetcode.com/discuss/interview-question/815454/Amazon-OA-question-or-SDE-1
- https://leetcode.com/problems/largest-rectangle-in-histogram/
- **Must do Dynamic programming Category wise:** https://leetcode.com/discuss/general-discussion/1050391/Must-do-Dynamic-programming-Problems-Category-wise


# Greedy Algorithm:
## 1. A Scheduling Application
**Setup:**
- One shared resource (e.g processor)
- Many jobs to do (e.g. processes)

**Question**: In what order should we sequence the jobs?
**Assume**
  - weight wj (priority)
  - lenght Lj

### Completion Times
**Definition:** The completion time Cj of job j = Sum of job lenghts up to and including j.

### The Objective Function
- There are differnt objective functions can be thought for the scheduling problem, here we assume:
  
**Goal: Minimize the weighted sum of completion times:**
```sh
min(sigma(wj* Cj))
```
### Intuition for Algorithm
**Recall:** want to minimize `min(sigma(wj* Cj))`

**Goal:** devise correct greedy algorithm

**Questions:** 
1. With equal lengths, schedule larger or smaller weight jobs earlier?
2. With equal weights, schedule shorter or longer jobs?

    1. larger weight jobs (more important jobs)
    2. shorter jobs (complete more jobs first)

### Resolving Conglicting Advice
**Questions:** what if wi > wj but Li > Lj?

**Idea:** assign **score** to jobs (instead of two params weight and duration) that are:
- increasing in weight
- decreasing in lenght
**Guess 1**: Order jobs by deceasing `wj - Lj`
**Guess 2**: Order jobs by decreasing `wj/Lj`

### Breaking a Greedy Algorithm
- to distinguish guess 1 and 2: find example where the two algorithms produces different outputs (At least one will be incorrect):
- Example:
  - `L1 = 5, w1 = 3` (longer ratio)
  - `L1 = 2, w1 = 1` (larger difference)

**Question**
What is the sum of weighted completion times of algorithms 1 and 2 respectively? 23, 22

**So far:** Algo 1 not always correct
**Claim:** ALgo 2 is always correct (not obvious!)
**Running time:** `O(n log n)` **just need to sort**


## 2. MST (Minimum Spanning Tree)
**Informal Goal**: connect a bunch of points together as cheaply as possible

**Application:** Clustinering, networking

**Blazingly fast Greedy Algorithms**:
  - Prim's Algorithm
  - Kruskal's Algorithm
=> _O(m log n)_ m: # of edges, n: # ofnodes

### Problem Definition

**Input:** **Undirected** graph G = (V, E) and a cost `ce` for each edge `e (memeber of) E`
  - Assume adjacency list representation
  - Ok if edge cost are negative

**Output:** **minimum cost (some of edge costs) spanning tree** T that **spans all vertices**
1. T has no cycles
2. the subgraph (V,T) is **connected (contains path between each parit of vertices)**

**Assumption 1** : Input graph G is connected
- Else no spanning trees
- Easy to check in preprocessing (e.g., depth-first-search)

**Assumption 2**: Edge costs are distinct
- Prim + Kruskal remain correct with ties (which can be broken arbitrarily)

### Prim's MST Algorithm
- Initialize X = {s} (s is memebr of V vertices chosen arbitraty)
- T = 0 [initialize our spanning tree,  X = vertices spanned by tree-so-far T]
- While X != V (while x does not include all veritices):
  - Let e = (u, v) be the cheapest edge of G with u is part of X and v is not part of X
  - Add e to T
  - Add v to X
While loop: Increase # of spanned veritces in cheapest way possible.

This algorithm time complexity:
- O(n) iteration (n: number of veritces)
- O(m) time per iteration (m of edges)

  ==> _O(mn)_

### Prim's MST Algorithm with Heaps
- The raison d'etre of a heap is to speed up **repeated minimum computations**
- heap supports insert, extract with min key value, and delete in _O(log n)_ where n is number of objects in the heap

**idea 1:** use heap to store edges, with keys = edge costs `O(mlog n)` of Prim's algorithm
**idea 2:** store vetices in the heap instead of the edges (better):

**Invariant 1**: Elements in the heap = vertices of V - X (the vertices that we haven't spanned yet)

**Invariant 2**: for v member of V-X, `key[v]= cheapest edge (u,v) with u member of X`. If the key `v` does not have any edges that goes to X, we define the edge as Inf.

**1. How to setup heap at the start of Prim's algorithm so both of those Invariants are satisfied?**
- At the start `X` contains an arbitraty vertex `s` and `V-X` contains other vertices other than `s`
- The key value of the heap is the cheapest edge between that each vertex (V-X) and `s` if there is one or Inf if there is none.
- So with a **single scan through the edges `O(m)`** we can compute the key values for each vertex that needs to go on heap and then we need to **insert those `n-1` vertices into heap: `O(n log n)`**
- So we can initialize heap with `O(m + nlogn) = O(m log n)` time perprocessing. The reason we have `O(m + nlogn) = O(m log n)`  is that:
  - `m >= n - 1` since the graph is connected, so we can replace `n` with `m` and factor out `m`.


Here's a basic outline of how Prim's algorithm works:

1. Start with an arbitrary vertex, setting its key value to 0 (the cost of adding that vertex to X spanning tree) and all others to infinity.
2. Add this vertex to the MST (X spanning tree).
3. For each vertex adjacent to a vertex in the MST, update its key value to the minimum of its current key value and the cost of the edge connecting it to the MST:
- When v (from V - X) added to X:
  - If each edge (v, w):
    - If w is a member of V - X -> **The only vertex whose key might have changed since the cut boundary is changed**
      1. Delete w from heap (at position i and not from the top - depending on where `w` is on the heap, we need some book-keeping)
      2. Recompute `key[w] := min{key[w], cvw}`
      3. Re-Insert into heap

4. Select the vertex with the smallest key value not in the MST and add it to the MST.
5. Repeat steps 3 and 4 until all vertices are in the MST.

### Running Time with Heaps
- n - 1 inserts during preprocessing
- n - 1 extract-mins (one per iteration of while loop)
- Each edge (v,w) triggers one Delete/Insert combo (**when its first endpoint is sucked into X**)
- => `O(m)` heap operation (**recall m >= n - 1 since graph is connected**)
- => `O(m log n)` time




## Memoization
- Interleaving string (https://leetcode.com/problems/interleaving-string/). 
- Explanation: https://www.youtube.com/watch?v=EzQ_YEmR598
- We use sliding pointer technique. We use 3 pointers for s1, s2 and s3 strings.
1. if s1 isi divided into N substrings, s2 should be divided into N or N-1 or N+1 substrings
2. len(s3) == len(s1) + len(s2)
3. The order of substrings should be maintained in s3 (alternating)
4. The number of unique characters in s3 should be equal to number of that character in s1 + s2
5. With sliding poniter technique, we define 3 pointes for each s1, s2, s3 strings and when p3 moves on each character in s3, there are 2 options: it can belong to s1 or s2. and so we can create a tree of choices starting from the first character of s3 as the root and each node has two children (each edge corresponds to s1 and s2).
6. With this approach, the time complexity: since each postion has two option times the number of chars in s1 and s2: 2^(m+n). The space complexity is from the recursion call tree depth == len(s3) == m+n
7. to improve it, we can use memoization to remember and save the result of a calculation and for that we use map to save key-value pairs. keys should be the unique states and so one way to create unique keys from three pointers p1, p2 and p3: key = p1 + * + p2 + * + p3 and the value will be whether 

- In memoization version of pascal number or fibonacci, the **Time complexity is O(N) and the space complexity if O(N)**. Without memoization, the time complexity would be **O(2^N)**.


## Regular Expression Matching (https://leetcode.com/problems/regular-expression-matching/)
- (https://www.youtube.com/watch?v=l3hda49XcDE)

```py
def isMatch(s,p):
    
    dp = [[False]*(len(p) + 1) for _ in range(len(s) + 1)]
    dp[0][0] = True
    
    # Deals with edge cases like a*, a*b*, a*b*c* which is correct for an empty string s:''
    for j in range(1, len(p)+1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-2]
            
            
    for i in range(1, len(s) + 1):
        for j in range(1, len(p) + 1):
            # matching characters or .
            if p[j-1] == s[i-1] or p[j-1] == '.':
                dp[i][j] = dp[i-1][j-1]
                
            elif p[j-1] == '*':
                # consider either zero occurance or one or more occurances
                dp[i][j] = dp[i][j-2] or dp[i][j-1]
                
                if p[j-2] == s[i-1] or dp[j-2] == '.':
                    dp[i][j] = dp[i-1][j] or dp[i][j]
                                
            else:
                dp[i][j] = False
                   
    return dp[-1][-1]
```

## Leetcode: Stone Game (https://leetcode.com/problems/stone-game/)
- https://www.youtube.com/watch?v=YVmTjwBjYwo, we define a helper function which returns the difference bw Alice's total stone and Bob's total stones. The helper function gets `i` and `j` as the left and right indices of the stone array. If's a recursive function and when `i > j` then the recursive finishes. So Alice has two options: she can either choose the first element or the last one:
    - If she takes p[i:left], the difference between Alice's and Bob's total stones are: `p[i] - helper[i + 1, j]`
    - If she takes p[j:right], the difference between Alice's and Bob's total stones are: `p[j] - helper[i, j - 1]`
    - We want to find the max between those two options for Alice.
    - The `helper[i+1, j] or helper[i, j-1]` is Bob's pick.

So if the return result from `helper[0, len(p) - 1] > 0` it means Alice wins otherwise Bob wins. Since some of those indices are repeated, we use memoization:

```py
def stoneGame(piles):
    """
    :type piles: List[int]
    :rtype: bool
    """
    memo = dict()
    def helper(i,j):
        if i > j: return 0

        if (i,j) in memo:
            return memo[(i,j)]
        else:
            leftPlay = piles[i] - helper(i+1, j)
            rightPlay = piles[j] - helper(i, j-1)

        memo[(i,j)] = max(leftPlay, rightPlay)
        return memo[(i,j)]


    n = len(piles)
    return helper(0, n-1) > 0

```

