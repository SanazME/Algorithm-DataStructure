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
