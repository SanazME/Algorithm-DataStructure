## BackTracking

- Recall that to implement backtracking:
  
1. we implement a backtrack function that makes some changes to the state,
2. calls itself again,
3. and then when that call returns it undoes those changes (this last part is why it's called "backtracking").

- https://leetcode.com/explore/featured/card/recursion-ii/472/backtracking/2804/

- Conceptually, one can imagine the procedure of backtracking as the tree traversal. Starting from the root node, one sets out to search for solutions that are located at the leaf nodes. Each intermediate node represents a partial candidate solution that could potentially lead us to a final valid solution. At each node, we would fan out to move one step further to the final solution, i.e. we iterate the child nodes of the current node. Once we can determine if a certain node cannot possibly lead to a valid solution, we abandon the current node and backtrack to its parent node to explore other possibilities. It is due to this backtracking behaviour, the backtracking algorithms are often much faster than the brute-force search [2] algorithm, since it eliminates many unnecessary exploration.

## Backtacking Template
```py
def backtrack(candidate):
    if find_solution(candidate):
        output(candidate)
        return
    
    # iterate all possible candidates.
    for next_candidate in list_of_candidates:
        if is_valid(next_candidate):
            # try this partial candidate solution
            place(next_candidate)
            # given the candidate, explore further.
            backtrack(next_candidate)
            # backtrack
            remove(next_candidate)
```
Here are a few notes about the above pseudocode.

- Overall, the enumeration of candidates is done in two levels:
    1). at the first level, the function is implemented as recursion. At each occurrence of recursion, the function is one step further to the final solution.
    2). as the second level, **within the recursion, we have an iteration** that allows us to explore all the candidates that are of the same progress to the final solution.

- The **backtracking should happen at the level of the iteration within the recursion**. 

- Unlike brute-force search, in backtracking algorithms we are often able to determine if a partial solution candidate is worth exploring further (i.e. `is_valid(next_candidate)`), which allows us to prune the search zones. This is also known as the constraint, e.g. the attacking zone of queen in N-queen game. 

- There are two symmetric functions that allow us to **mark the decision** (`place(candidate)`) and **revert the decision** (`remove(candidate)`).  

## Square Matrice Properties N * N:
1. Primary diagonal : all points wher `i = j`.
2. Anti-diagonal: all points where `j = N - i - 1` (start from top-right and going to left-bottom)
3. For each node (i,j):
  - its primary diagonal nodes: `(k, l) where: k - l = i - j`
  - its secondary diagonal node: `(k, l) where:  k + l = i + j`

## 1. Queen II
- https://leetcode.com/problems/n-queens-ii

**Intuition**

We can still follow the strategy of generating board states, but we should never place a queen on an attacked square. This is a perfect problem for backtracking - place the queens one by one, and when all possibilities are exhausted, backtrack by removing a queen and placing it elsewhere.

If you're not familiar with backtracking, check out the backtracking section of our Recursion II Explore Card.

Given a board state, and a possible placement for a queen, we need a smart way to determine whether or not that placement would put the queen under attack. A queen can be attacked if another queen is in any of the 4 following positions: on the same row, on the same column, on the same diagonal, or on the same anti-diagonal.

**Recall that to implement backtracking, we implement a backtrack function that makes some changes to the state, calls itself again, and then when that call returns it undoes those changes (this last part is why it's called "backtracking").**

Each time our `backtrack` function is called, we can encode state in the following manner:

- To make sure that we only place 1 queen per **row**, we will pass an integer argument `row` into `backtrack`, and will only place one queen during each call. Whenever we place a queen, we'll move onto the next row by calling `backtrack` again with the parameter value `row + 1`.

- To make sure we only place 1 queen per **column**, we will use a set. Whenever we place a queen, we can add the column index to this set.

The diagonals are a little trickier - but they have a property that we can use to our advantage.

- For each square on a given diagonal, the difference between the row and column indexes `(row - col)` will be constant. Think that moving diagonally, increase row by 1 and increase col by 1 as well so the diff is always constant.

- For anti-diagonal, the sum of col and column indexes will be constant: `(row + col)`. Moving anti-diagonally, increases row by 1 and decreases col by 1 so the sum of indexes will stay constant.

**Algorithm**
We'll create a recursive function `backtrack` that takes 4 arguments to maintain the board state. The first parameter is the row we're going to place a queen on next, and the other 3 are sets that track which columns, diagonals, and anti-diagonals have already had queens placed on them. The function will work as follows:

1. If the current row we are considering is greater than n, then we have a solution. Return 1.

2. Initiate a local variable `solutions = 0` that **represents all the possible solutions that can be obtained from the current board state.**

3. Iterate through the columns of the current row. At each column, we will attempt to place a queen at the square `(row, col)` - remember we are considering the current row through the function arguments.

  - Calculate the diagonal and anti-diagonal that the square belongs to. If there has been no queen placed yet in the column, diagonal, or anti-diagonal, then we can place a queen in this column, in the current row.

  - If we can't place the queen, skip this column (move on to try with the next column).

4. If we were able to place a queen, then update our 3 sets (cols, diagonals, and antiDiagonals), and call the function again, but with row + 1.

5. The function call made in step 4 explores all valid board states with the queen we placed in step 3. Since we're done exploring that path, backtrack by removing the queen from the square - this just means removing the values we added to our sets.

```py
class Solution:
    def totalNQueens(self, n):
        def backtrack(row, diagonals, anti_diagonals, cols):
            # Base case - N queens have been placed
            if row == n:
                return 1

            solutions = 0
            for col in range(n):
                curr_diagonal = row - col
                curr_anti_diagonal = row + col
                # If the queen is not placeable
                if (col in cols 
                      or curr_diagonal in diagonals 
                      or curr_anti_diagonal in anti_diagonals):
                    continue

                # "Add" the queen to the board
                cols.add(col)
                diagonals.add(curr_diagonal)
                anti_diagonals.add(curr_anti_diagonal)

                # Move on to the next row with the updated board state
                solutions += backtrack(row + 1, diagonals, anti_diagonals, cols)

                # "Remove" the queen from the board since we have already
                # explored all valid paths using the above function call
                cols.remove(col)
                diagonals.remove(curr_diagonal)
                anti_diagonals.remove(curr_anti_diagonal)

            return solutions

        return backtrack(0, set(), set(), set())
```

**Time Complexity: O(N!)**
- unlike brute-force, we don't place queen on every tile, the first row hsa `N` option, the second has `N-1`, the third has `N-2` etc: `N*(N-1)*(N-2)*...1 = N!`

**Space Complexity: O(N)**  
- `N` is the number of queens (which is the same as the width and height of the board)
- Extra memory used includes the 3 sets used to store board state as well as **the recursion call stack** . All of this scale linearly with the number of queens

## Queen II
- Another solution: https://www.educative.io/courses/mastering-algorithms-for-problem-solving-in-python/n-queens is to:
- define an array `Q: [0] * N` where here `Q[i]` `i` is the row i and `Q[i]` stores the index of a column where the queen is located in row i.
- For each row, we have two for loops:
  - we iterate throught column
  - for a given column, we iterate through all previous rows (0 -> row - 1) and check if the queen has been placed in previous rows in the same column, in the diagonal to that node or in anti-diagonal to that node:
```py
def totalNQueens(self, n: int) -> int:
    def place_queen(Q, row):
            if row == len(Q):
                self.count += 1
                print(self.count)

            else:
                for col in range(len(Q)):
                    legal = True
                    for i in range(row):
                        if Q[i] == col or (Q[i] == col - row + i) or (Q[i] == col + row - i):
                            legal = False
                    if legal:
                        Q[row] = col
                        place_queen(Q, row + 1)


        place_queen([0]*n, 0)

        return self.count
```

## Beautiful Arrangement
- https://leetcode.com/problems/beautiful-arrangement/description/
```py
class Solution(object):
    def countArrangement(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n <=1 :
            return n

        self.count = 0
        self.backTracking(n, 1, [])

        return self.count

    def backTracking(self, n, position, tmp):
        if len(tmp) == n:
            print(tmp)
            self.count += 1
            return

        for num in range(1, n+1):
            if num not in tmp and (num % position == 0 or position % num == 0):
                tmp.append(num)
                self.backTracking(n, position + 1, tmp)
                # print(tmp)
                tmp.pop()
# OR
class Solution(object):
    def countArrangement(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n <=1 :
            return n

        self.count = 0
        visited = [False] * (n+1)
        self.backTracking(n, 1, visited)

        return self.count

    def backTracking(self, n, position, visited):
        if position > n:
            self.count += 1
            return

        for num in range(1, n+1):
            if visited[num] == False and (num % position == 0 or position % num == 0):
                visited[num] = True
                self.backTracking(n, position + 1, visited)
                visited[num] = False
```
