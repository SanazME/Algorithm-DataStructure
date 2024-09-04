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
