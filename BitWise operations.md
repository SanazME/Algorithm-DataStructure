- https://realpython.com/python-bitwise-operators/
- Subtraction in binary:
```py
1 - 1 = 0
0 - 0 = 0
1 - 0 = 1
0 - 1 = 1 (borrow 1)
```
- `n &= n - 1` remove the rightmost in binary representation. for example:
```sh
n = 10110100
n - 1 = 10110011

n & (n - 1) = 10110100 & 10110011 = 10110000
```
- https://leetcode.com/problems/number-of-1-bits/
```py
class Solution:
    def hammingWeight(self, n: int) -> int:
        ones = 0
        
        while n:
            n &= n - 1
            ones += 1
            
        return ones
```
(XOR) Exclusive Or
- a ^ a = 0
- a ^ 0 = a
- a ^ a ^ b ^ c ^ c = b
- https://leetcode.com/problems/single-number/
