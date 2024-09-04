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




