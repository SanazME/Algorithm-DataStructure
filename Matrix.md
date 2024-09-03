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
