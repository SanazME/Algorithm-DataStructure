## Important conversions their meaning
- The number `a` is always the number that we try to shift it and the number on the right is how much offset we want to do:
- `a << 1` : moves the bits of its first operand to the **left** by the **number of places specified in its second operand**. It also takes care of inserting enough zero bits to fill the gap that arises on the **right edge** of the new bit pattern 
- `a >> 1` : moves the bits of its first operand to the **right** by the **number of places specified in its second operand**. It also takes care of inserting enough zero bits to fill the gap that arises on the **left edge** of the new bit pattern and **The rightmost bits always get dropped**.


### 1. Converting a list of bytes to an n-bit integer:
This conversion is used when you have a list of bytes (each byte representing an 8-bit value) and you want to combine them into a single n-bit integer. 
```py
# most significant byte + .... + least significant byte
(l[0] << (n-1)*8) + (l[1] << (n-2)*8) + ...
```
Suppose we have a list of 4 bytes: `[0xAA, 0xBB, 0xCC, 0xDD]`. We want to convert this to a 32-bit integer (n = 4 bytes = 32 bits).
The calculation would be:
```py
(0xAA << 24) + (0xBB << 16) + (0xCC << 8) + 0xDD
```
This is equivalent to:
```py
0xAABBCCDD
```
In decimal, this would be `2864434397`.

The alternative method using `next()` is useful when you're working with an iterator:
```py
l = iter([0xAA, 0xBB, 0xCC, 0xDD])
result = (next(l) << 24) + (next(l) << 16) + (next(l) << 8) + next(l)
```
These 24, 16, and 8 numbers comes from the position of each byte in the final 32-bit integer:
- we're working with a 32-bit integer, which is 4 bytes.
- each byte is 8 bits.
- we want to place each byte in its correct position within the 32-bit integer.
For our example:
- 0xAA << 24: This shifts 0xAA left by 24 bits, placing it in the most significant byte position.
- 0xBB << 16: This shifts 0xBB left by 16 bits, placing it in the second most significant byte position.
- 0xCC << 8: This shifts 0xCC left by 8 bits, placing it in the third byte position.
- 0xDD: This doesn't need to be shifted as it's already in the least significant byte position.
```sh
0xAA << 24 = 0xAA000000
0xBB << 16 =   0xBB0000
0xCC << 8  =     0xCC00
0xDD       =       0xDD
--------------------
           0xAABBCCDD (when added together)
```

### 2. Converting an n-bit integer to a list of bytes:
This is the reverse operation of the previous one. Given a 32-bit integer, we want to extract each byte. The formula you provided is:
```py
[(number >> 24) & 255, (number >> 16) & 255, (number >> 8) & 255, number & 255]
```

Let's use the same example: 0xAABBCCDD (2864434397 in decimal)
```sh
(number >> 24) & 255: This extracts 0xAA
(number >> 16) & 255: This extracts 0xBB
(number >> 8) & 255: This extracts 0xCC
number & 255: This extracts 0xDD
```
The result would be [0xAA, 0xBB, 0xCC, 0xDD] or [170, 187, 204, 221] in decimal.

Let's break down the operation (number >> 24) & 255 using 0xAABBCCDD as an example:

1. First, number >> 24: 10xAABBCCDD >> 24 = 0x000000AA`. This right-shift brings the most significant byte (0xAA) to the least significant position.
2. Then, & 255: `0x000000AA & 0x000000FF = 0x000000AA`. The bitwise AND operation with 255 (0xFF) ensures that even if there were any remaining 1 bits in the upper 24 bits (which there aren't in this case due to the right shift), they would be set to 0.
```sh
00000000 00000000 00000000 10101010  (0x000000AA after right shift)
& 00000000 00000000 00000000 11111111  (0xFF or 255)
-------------------------------------
  00000000 00000000 00000000 10101010  (Result: 0xAA)
```

0.1. When you need to convert a n-bit integer number to its n//8 byte representations: for example convert a 32-bit integer into a list of 4 byte integers:

- `n >> 24`: This operation shifts the bits of the number `n` 24 places to the right. The vacant bit positions are filled with zeros if `n` is a non-negative number and with ones if `n` is a negative number.
- `(n >> 24) & 255` : The result of the previous operation is then bitwise ANDed with 255. This operation retains **the 8 least significant bits of the result n >> 24** and sets all other bits to 0. The number 255 is represented in binary as **11111111** so when bitwise AND is performed with this number, it keeps the 8 least significant bits.

Here's what each part does:

1. `number >> 24 & 255`: This operation shifts the bits of number 24 places to the right and then performs a bitwise AND operation with 255. This effectively extracts the most significant 8 bits of number. 255 is `11111111` so it masks and limits the s

2. number >> 16 & 255: This operation shifts the bits of number 16 places to the right and then performs a bitwise AND operation with 255. This effectively extracts the second most significant 8 bits of number.

number >> 8 & 255: This operation shifts the bits of number 8 places to the right and then performs a bitwise AND operation with 255. This effectively extracts the third most significant 8 bits of number.

number & 255: This operation performs a bitwise AND operation between number and 255. This effectively extracts the least significant 8 bits of number.

### 3. Finding the index of the least significant set bit:

First, let's clarify what we mean by "least significant set bit":

- In binary representation, it's the **rightmost bit that is 1**.
- The index starts from 0, counting from right to left.

For example, in the binary number 1010100:
- The least significant set bit is the rightmost 1.
- Its index is 2 (counting from right, starting at 0).

To find the index of the least significant set bit in the binary representation of a the **input integer x**, to find the position of the first 1 bit starting from the bit at position 0:

- `1 << i` where i is the location of the least significant set bit, the left shift calculated the value of that bit: 1, 2, 4, 8, ... 

```py
# x is input integer but with & we perform binary AND operation
# 1 << i (2**i), shifts 1 with i and everytime we check if the result of x AND (that bit) is 0 or 1

for i in range(32):
    if x & (1 << i):
        return i

# OR
x & (-x)


(1 << i) produces: 1, 2, 4, 8, etc
```

2.
  -  Let's say we have a 32-bit integer where each group of byte (8 bits) is printed as a decimal number with `.`: `15.136.255.107` : this is IP address
  -  CIDR block is a format used to denote a specific set of IP addresses. It is a string consisting of a base IP address, followed by a slash, followed by a prefix length k. The addresses it covers are all the IPs whose **first k bits** are the same as the base IP address.
  -  For example, `"123.45.67.89/20"` is a CIDR block with a prefix length of 20. Any IP address whose binary representation matches `01111011 00101101 0100xxxx xxxxxxxx`, where x can be either 0 or 1, is in the set covered by the CIDR block.
  
  -  To convert a IP address to it its 32-bit integer (in base 10):
```py
# given
ip = [255, 0, 0, 7]

# the number representation
number = (ip[0] << 24) + (ip[1] << 16) + (ip[2] << 8) + (ip[3] << 0)

# number = 4278190087 which is equivalent to : 255 * 2^24 + 0 * 2^16 + 0 * 2^8 + 7 
```
`ip` : list as a byte in a 4-byte (32-bit) integer
`ip[0] << 24`: This operation takes the firt number in the list, treats it as the most significant byte and shifts it left by 3 * 8 = 24 bits. It is equivalent to : `ip[0] * 2^24 + ip[1] * 2^16 + ...`
...
All these values are then added together to **form a singel integer**


4. `&` it is bitwise AND operations:
```py
(10 & 7)  # returns 2
# 10  1010
# 7   0111
```

### IP to CIDR
- https://leetcode.com/problems/ip-to-cidr/description/

**Solution**
What we need to do is to find the minimum cover(n) starting from the given ip. The main idea is that calculating the n next ips in the form of number(11111111 00000000 00000000 00000111) instead of ip(e.g. 255.0.0.7). Plus, as for 255.0.0.7/x, x indicates 2^(32-x) can be covered.

ip2number(self,ip) helps us get the corresponding number(32bit) of ip, so we know where to start.

Then, number2ip(self,n) helps us to get the ip form.

ilowbit(self,x) would return the index i of the first '1' starting from the right. The index of the first 1 helps to define how many ips have the same (32-i)part. For example, 255.0.0.8 - 255.0.0.15 have the same part 11111111 00000000 00000000 00001xxx.

The rest thing is just countdown from the given ip. For example, 255.0.0.7/32 (must start from this) is the first ip because 255.0.0.7/31 has 255.0.0.6 when the 32th bit can be 0 or 1. Then, 255.0.0.8/29 has the last 3bits can be 000 to 111. Now we get 9 ips, after adding 255.0.0.16/32, we get the answer.

```py
class Solution:
    def ipToCIDR(self, ip: str, n: int) -> List[str]:
        if n == 1:
            return [ip + "/32"]
    
        number = self.ipToNumber(ip)
        result = []
        
        while n:
            idxLeastBit, leastBit = self.leasSignificanttBit(number)
            
            while leastBit > n:
                leastBit /= 2
                idxLeastBit -= 1
                
            result.append(self.numberToIp(number, idxLeastBit))
            
            n -= leastBit
            number = int(number + leastBit)
            
        return result


    def ipToNumber(self, ip):
        # segments = list(map(lambda x: int(x), ip.split(".")))
        # number = (segments[0] << 24) + (segments[1] << 16) + (segments[2] << 8) + (segments[3] << 0)
        # OR:
        segments = map(lambda x: int(x), ip.split("."))
            
        number = (next(segments) << 24) + (next(segments) << 16) + (next(segments) << 8) + (next(segments) << 0)

            
        return number

    def leasSignificanttBit(self, number):
        # 32-bit number
        for i in range(32):
            if (number & (1 << i)):
                return (i, 1 << i)

        return (32, 1 << 32)

    
    def numberToIp(self, number, idxLeastBit):
        # 32 - IdxLeastBit
        baseIp = [str(number >> 24&255), str(number >> 16&255), str(number >> 8&255), str(number&255)]

        return '.'.join(baseIp) + "/" + str(32 - idxLeastBit)
```   

### bit shifting
The expression `MaxInt uint64 = 1<<64 - 1` in Go sets `MaxInt` as the maximum representable value for a `uint64` (unsigned 64-bit integer).

Breaking it down:

- `1<<64`: This is a bitwise left shift operation. It takes the binary representation of 1 (`000...001`) and shifts it to the left 64 times, effectively placing a 1 at the 65th bit position. Note that Go uses zero-based bit numbering, so this would be the bit at index 64 if we start counting from the rightmost bit as index 0.
  
- `1<<64 - 1`: The `- 1` then flips that 65th bit back to 0 and sets all the 64 bits to the right of it to 1.

In binary representation, `MaxInt` will be `111...111` (64 ones).

The reason for this is that `uint64` is an unsigned 64-bit integer, meaning it can represent integers from \(0\) to \(2^{64} - 1\), inclusive. \(2^{64} - 1\) is the largest integer that can be represented by 64 bits when all bits are set to 1.


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

