### Valid Sudoku
- https://leetcode.com/problems/valid-sudoku/description/?envType=study-plan-v2&envId=top-interview-150

- we encode what we see in row , col and block and if we hit those we return false:
```py
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        seen = set()

        for i in range(9):
            for j in range(9):
                number = board[i][j]

                if number != '.':
                    row_check = f"{number} in row {i}"
                    col_check = f"{number} in column {j}"
                    block_check = f"{number} in block {i//3}-{j//3}"

                    if (row_check in seen or 
                    col_check in seen or
                    block_check in seen):
                        return False
                   
                    seen.update([row_check, col_check, block_check])

        return True
```

### Spiral Matrix
- similar to soution for spiral matrix II, we move into one of 4 direction till we go out of bound or till we already seen that element:
```py
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        result = []
        rows = len(matrix)
        cols = len(matrix[0])
        
        """
        1: (0,1)
        2: (1,0)
        3: (0, -1)
        4: (-1, 0)
        """
        
        def outbound(i, j, rows, cols):
            return (i >= rows or i < 0) or (j >= cols or j < 0)
        
        dirList = [(0,1), (1,0),(0,-1),(-1,0)] 
        dirVal = 0
        di, dj = dirList[dirVal]
        i, j = 0, 0
        seen = set()
        count = 1
        
        while count <= rows * cols:
            num = matrix[i][j]
            seen.add((i,j))
            result.append(num)
            
            nexti, nextj = i + di, j + dj
            
            if outbound(nexti, nextj, rows, cols) or (nexti, nextj) in seen:
                dirVal = (dirVal + 1) % 4
            
            di, dj = dirList[dirVal]
            
            i += di
            j += dj
            
            count += 1
        
        return result
```


### Spiral Matrix II
- https://github.com/SanazME/Algorithm-DataStructure/blob/master/AmazonQuestions.md#13-spiral-matrix-ii


### Rotate Image
- https://leetcode.com/problems/rotate-image/description
- Let's consider a step-by-step rotation of a single element:

Think about the rotation in terms of layers. For an n x n matrix, you can divide the rotation into (n+1)/2 layers (rounded down). Start from the outermost layer and work your way inwards.
For each layer:

1. Consider four elements at a time - one from each corner of the current layer.
2. Perform a 4-way swap of these elements to rotate them clockwise.
3. Move to the next set of four elements in the layer.

he key to this solution is understanding the index transformations:

`[i][j] → [j][n-1-i] → [n-1-i][n-1-j] → [n-1-j][i] → [i][j]`

1. Start with position (i, j)
2. After a 90-degree clockwise rotation:
```sh
Before rotation:     After rotation:
  0 1 2                2 1 0
0 * - -              0 - - *
1 - - -              1 - - -
2 - - -              2 - - -
```
Here, * represents our element at (i, j) = (0, 0)
Notice that:
1. The element's distance from the top (i = 0) becomes its distance from the right side (j = 2 = n-1).
2. The element's distance from the left side (j = 0) becomes its distance from the top (i = 0).

This is why:
- i becomes the new j (distance from left becomes distance from top)
- j becomes n-1-i (distance from top becomes distance from right)
```py
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        
        for i in range(n//2):
            for j in range(i, n - 1 - i):
                tmp = matrix[i][j]

                matrix[i][j] = matrix[n - 1 - j][i]
                matrix[n - 1 - j][i] = matrix[n - 1 - i][n - 1 - j]
                matrix[n - 1 - i][n - 1 - j] = matrix[j][n - 1 - i]
                matrix[j][n - 1 - i] = tmp

```

## Set Matrix Zeros
- https://leetcode.com/problems/set-matrix-zeroes/
Algorithm Explanation:
- a. First, we check if the first row and first column contain any zeros. We store this information in first_row_zero and first_col_zero variables.
- b. We then use the first row and first column as markers. For any cell matrix[i][j] that is 0, we set matrix[i][0] and matrix[0][j] to 0.
- c. After marking, we iterate through the matrix again (except the first row and column) and set cells to 0 based on the markers in the first row and column.
- d. Finally, we handle the first row and column based on the first_row_zero and first_col_zero variables we set earlier.

Space; `O(1)`
Time: `O(nm)`
```py
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        m, n = len(matrix), len(matrix[0])
        first_row_zero = False
        first_col_zero = False
        
        # Check if the first row contains zero
        for j in range(n):
            if matrix[0][j] == 0:
                first_row_zero = True
                break
        
        # Check if the first column contains zero
        for i in range(m):
            if matrix[i][0] == 0:
                first_col_zero = True
                break
        
        # Use first row and column as markers
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == 0:
                    matrix[i][0] = 0
                    matrix[0][j] = 0
        
        # Set zeros based on markers
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0
        
        # Set first row to zero if needed
        if first_row_zero:
            for j in range(n):
                matrix[0][j] = 0
        
        # Set first column to zero if needed
        if first_col_zero:
            for i in range(m):
                matrix[i][0] = 0
```

## Game of Life
- https://leetcode.com/problems/game-of-life/
- we need to track both current and the next state in each cell and for that we can use bit manipulation. we will have the following combinations:
```sh
next start - current state
00  : dead now and dead later
10  : dead now and alive later
01  : live now and dead later
11  : live now and live later
```

```py
class Solution:
    def gameOfLife(self, board: List[List[int]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """

        n, m = len(board), len(board[0])
        
        def getNeighboarSum(i, j):
            neighbors = [(i-1, j-1), (i-1,j), (i-1,j+1),
            (i, j-1), (i, j+1),
            (i+1, j-1), (i+1, j), (i+1, j+1)]

            result = 0
            for row, col in neighbors:
                if 0 <= row < n and 0 <= col < m:
                    result += board[row][col] & 1

            return result
            

        for i in range(n):
            for j in range(m):
                countNeighbors = getNeighboarSum(i, j)
                if board[i][j] & 1:
                    if countNeighbors in [2,3]:
                        board[i][j] = 3 # 11
                    else: 
                        board[i][j] = 1  # 01
                    
                else:
                    if countNeighbors == 3:
                        board[i][j] = 2  # 10

            
        for i in range(n):
            for j in range(m):
                board[i][j] = board[i][j] >> 1

```
