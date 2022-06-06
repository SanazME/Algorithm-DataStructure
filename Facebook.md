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
