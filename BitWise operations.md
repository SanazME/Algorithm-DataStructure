- https://realpython.com/python-bitwise-operators/
- Subtraction in binary:
```py
1 - 1 = 0
0 - 0 = 0
1 - 0 = 1
0 - 1 = 1 (borrow 1) or use two's complement of the number?
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

## Sum and Subtraction operations on positive integers x + y, x - y
- Sum operation can be broken down to :
1. `x ^ y` (without carry)
2. `(x & y) << 1` (carry)

So we keep doing the same operation till carry is 0:
```py
x = (x ^ y)
y = (x & y) << 1
```

- Subtract:
1. `x ^ y` (without carry)
2. `((~x) & y) << 1`

### Operations
- `x << y` : returns `x` with the bits shifted to the left by `y` places (and the new bits on the right are zero). This is the same as mulitplying `x` by `2**y`.
- `x >> y` : returns `x` with the bits shifted to the right by `y` places (and the new bits on the left are zero). This is the same as // `x` by `2**y`.
- `x ^ y` : XOR
- `x | y`: OR
- `x & y`: AND
- `~ x`: complement of `x`. the same as `-x-1`

## Tricks
1. Union: `x | y`
2. Intersection: `x & y`
3. Substraction: `x & ~y`
4. Check if the last bit is one or not: `x & 1`
5. `x & 1 = x`
6. Sum of two numbers without carry: `x ^ y`
7. Carry: `(x & y) << 1`
8. Remove right most bit: `x & (x - 1)`
9. To get part of the series that we want we use `x & 1`:
    - for example to get the last 4 of x:  `x & 1b1111`

10. 32 bit mask in hexadecimal: `0xffffffff` (8 fs)

## Two's complement of a numer
- It is used to represent negative numbers in binary.
- It is a way of representing signed integers (both positive and negative) in a fixed number of bits (like 32 bits)
- In two's complement representation, the leftmost bit (also known as the "sign bit") is used to represent the sign of the number. If the leftmost bit is 0, the number is positive. If the leftmost bit is 1, the number is negative. 
- To convert a positive number to its two's complement representation, you simply represent the number in binary as usual.  For example, the number 5 in binary is `0b0101`. 
- To represent -5 in two's complement, **for negative numbers: you first flip all the bits in the binary representation of the positive number (`0b0101` becomes `0b1010`), and then add 1 to the result (0b1010 + 1 = 0b1011)**. So the two's complement representation of -5 in 4 bits is 0b1011.

## Masks
- In Python unlike other languages the range of bits for representing a value is not 32, its much much larger than that. This is great when dealing with non negative integers, however this becomes a big issue when dealing with negative numbers ( two's compliment)

Why ?

Lets have a look, say we are adding -2 and 3, which = 1

In Python this would be ( showing only 3 bits for clarity )

1 1 0 +
0 1 1

Using binary addition you would get

0 0 1

That seems fine but what happended to the extra carry bit ? ( 1 0 0 0 ), if you were doing this by hand you would simply ignore it, but Python does not, instead it continues 'adding' that bit and continuing the sum.

1 1 1 1 1 1 0 +
0 0 0 0 0 1 1
0 0 0 1 0 0 0 + ( carry bit )

so this actually continues on forever unless ...

Mask !

The logic behind a mask is really simple, you should know that x & 1 = x right, so using that simple principle,

if we create a series of 4 1's and & them to any larger size series, we will get just that part of the series we want, so

1 1 1 1 1 0 0 1
0 0 0 0 1 1 1 1 &

0 0 0 0 1 0 0 1 ( Important to note that using a mask removes the two's compliment)


### Count number of ones in the binary representation of a given number:
**For two postive numbers**
1. Start from the least significant bit (right) and check if it's 1: `n & 1`
2. Do one bit shifted to the right and repeat the process: `n = n >> 1`
```py
def count_ones(n):
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count
```
**For negative numbers**
- For this question leetcode uses 32 bits, so you just need to create a 32 bit mask of 1's , the quickest way is to use hexadecimal and 0xffffffff, you can write the binary form if you prefer it will work the same.
- Note the final check, if b = 0 that means the carry bit 'finished', but when there is a negative number ( like -1), the carry bit will continue until it exceeds our 32 bit mask ( to end while loop ) it wont be 0 so in that case we use the mask.

```py
class Solution:
    def getSum(self, a: int, b: int) -> int:
        
        # 32 bit mask in hexadecimal
        mask = 0xffffffff
        
        # works both as while loop and single value check 
        while (b & mask) > 0:
            
            carry = ( a & b ) << 1
            a = (a ^ b) 
            b = carry
        
        # handles overflow
        return (a & mask) if b > 0 else a
```

