## 52. Find Median from a Data Stream
- You need to implement a data structure that will store a dynamically growing list of integers and provide efficient access to their median.
- **Solution**:
- We will assume that `x` is the median age of a user in a list. Half of the ages in the list will be smaller than (or equal to) `x`, and the other half will be greater than (or equal to) `x`. We can divide the list into two halves: one half to store the smaller numbers (say `smallList`), and one half to store the larger numbers (say `largeList`). The median of all ages will either be the largest number in the `smallList` or the smallest number in the `largeList`. If the total number of elements is even, we know that the median will be the average of these two numbers. The best data structure for finding the smallest or largest number among a list of numbers is a Heap.

Here is how we will implement this feature:

1. First, we will store the first half of the numbers (`smallList`) in a **Max Heap**. We use a **Max Heap** because we want to know the largest number in the first half of the list.

2. Then, we will store the second half of the numbers (`largeList`) in a **Min Heap**, because we want to know the smallest number in the second half of the list.

3. We can calculate the median of the current list of numbers using the top element of the two heaps.

- **Time Complexity**:
  - **Insert**: `O(logn)`
  - **Find Median**: `O(1)`
- **Space Complexity**: O(n)

```py
from heapq import *
class median_of_ages:

  maxHeap = []
  minHeap = []

  def insert_age(self, num):
    if not self.maxHeap or -self.maxHeap[0] >= num:
      heappush(self.maxHeap, -num)
    else:
      heappush(self.minHeap, num)

    if len(self.maxHeap) > len(self.minHeap) + 1:
      heappush(self.minHeap, -heappop(self.maxHeap))
    elif len(self.maxHeap) < len(self.minHeap):
      heappush(self.maxHeap, -heappop(self.minHeap))

  def find_median(self):
    if len(self.maxHeap) == len(self.minHeap):
      # we have even number of elements, take the average of middle two elements
      return -self.maxHeap[0] / 2.0 + self.minHeap[0] / 2.0

    # because max-heap will have one more element than the min-heap
    return -self.maxHeap[0] / 1.0


# Driver code

medianAge = median_of_ages()
medianAge.insert_age(22)
medianAge.insert_age(35)
print("The recommended content will be for ages under: " + str(medianAge.find_median()))
medianAge.insert_age(30)
print("The recommended content will be for ages under: " + str(medianAge.find_median()))
medianAge.insert_age(25)
print("The recommended content will be for ages under: " + str(medianAge.find_median()))
```

## 53 Happy Number
- https://leetcode.com/problems/happy-number/description/
- Based on our exploration so far, we'd expect continually following links to end in one of three ways.

It eventually gets to 111.
It eventually gets stuck in a cycle.
It keeps going higher and higher, up towards infinity.
That 3rd option sounds really annoying to detect and handle. How would we even know that it is going to continue going up, rather than eventually going back down, possibly to 1?1?1? Luckily, it turns out we don't need to worry about it. Think carefully about what the largest next number we could get for each number of digits is.

Digits	Largest	Next
1	9	81
2	99	162
3	999	243
4	9999	324
13	9999999999999	1053

For a number with 333 digits, it's impossible for it to ever go larger than 243243243. This means it will have to either get stuck in a cycle below 243243243 or go down to 111. Numbers with 444 or more digits will always lose a digit at each step until they are down to 333 digits. So we know that at worst, the algorithm might cycle around all the numbers under 243243243 and then go back to one it's already been to (a cycle) or go to 111. But it won't go on indefinitely, allowing us to rule out the 3rd option.

Algorithm

There are 2 parts to the algorithm we'll need to design and code.

Given a number nnn, what is its next number?
Follow a chain of numbers and detect if we've entered a cycle.
Part 1 can be done by using the division and modulus operators to repeatedly take digits off the number until none remain, and then squaring each removed digit and adding them together. Have a careful look at the code for this, "picking digits off one-by-one" is a useful technique you'll use for solving a lot of different problems.

Part 2 can be done using a HashSet. Each time we generate the next number in the chain, we check if it's already in our HashSet.

If it is not in the HashSet, we should add it.
If it is in the HashSet, that means we're in a cycle and so should return false.
The reason we use a HashSet and not a Vector, List, or Array is because we're repeatedly checking whether or not numbers are in it. Checking if a number is in a HashSet takes O(1)O(1)O(1) time, whereas for the other data structures it takes O(n)O(n)O(n) time. Choosing the correct data structures is an essential part of solving these problems.

```py
class Solution:
    def isHappy(self, n: int) -> bool:
        mem = set()
        
        def getNextNumber(num):
            total_sum = 0
            while num > 0:
                total_sum += (num % 10) ** 2
                num = num // 10

            return total_sum


        while n != 1 and n not in mem:
            mem.add(n)
            n = getNextNumber(n)
            
        return n == 1
```

- **Time complexity O(log n)**
  - In `getNextNumber` function, in while loop, we go down the number of digits in the number : `O(log n)` and operation in each step is constant.
  - In the main while loop, We determined above that once a number is below 243, it is impossible for it to go back up above 243. Therefore, based on our very shallow analysis we know for sure that once a number is below 243, it is impossible for it to take more than another 243 steps to terminate. Each of these numbers has at most 3 digits. With a little more analysis, we could replace the 243 with the length of the longest number chain below 243, however because the constant doesn't matter anyway, we won't worry about it.

For an n above 243, we need to consider the cost of each number in the chain that is above 243. With a little math, we can show that in the worst case, these costs will be `O(log⁡n)+O(log⁡log⁡n)+O(log⁡log⁡log⁡n)...` Luckily for us, the O(log⁡n)O(\log n)O(logn) is the dominating part, and the others are all tiny in comparison (collectively, they add up to less than log⁡n)\log n)logn), so we can ignore them.

- **Space complexity O(log n)**

Think about what would happen if you had a number with 1 million digits in it. The first step of the algorithm would process those million digits, and then the next value would be, at most (pretend all the digits are 9), be 81∗1,000,000=81,000,00081 * 1,000,000 = 81,000,00081∗1,000,000=81,000,000. In just one step, we've gone from a million digits, down to just 8. The largest possible 8 digit number we could get is 99,9999,99999,9999,99999,9999,999, which then goes down to 81∗8=64881 * 8 = 64881∗8=648. And then from here, the cost will be the same as if we'd started with a 3 digit number. Starting with 2 million digits (a massively larger number than one with a 1 million digits) would only take roughly twice as long, as again, the dominant part is summing the squares of the 2 million digits, and the rest is tiny in comparison.

## 54. Isomorphic Strings
- https://leetcode.com/problems/isomorphic-strings/editorial/
```py
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        map = {}  #map[s] -> t
        seen = set()
        for i in range(len(s)):
            if s[i] in map:
                if map[s[i]] != t[i]:
                    return False
            else:
                if t[i] in seen:
                    return False
                
                map[s[i]] = t[i]
                seen.add(t[i])
            
        return True
```

## 55. Reverse Words in a String III
- https://leetcode.com/problems/reverse-words-in-a-string-iii/description/

```py
class Solution:
    def reverseWords(self, s: str) -> str:
        count = 0
        final = ""

        while count < len(s):
            wordIdx = count
            word = ""
 
            while wordIdx < len(s) and s[wordIdx] != " ":
                word += s[wordIdx]
                wordIdx += 1
            
            final += self.reverseWord(word)
            count = wordIdx + 1

        return final.rstrip()

    
    def reverseWord(self, word):
        print(word)
        if word == "":
            return ""
        
        word_list = list(word)
        word_list = word_list[::-1]
        reversedWord = ''.join(word_list)

        return reversedWord + " "

```

## 56. Can Place Flowers
- https://leetcode.com/problems/can-place-flowers/description/
- we go through the list and check if the element is 0 and it's left and right neighbours are also zero...
```py
class Solution:
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:

        if n == 0: return True
        for i in range(len(flowerbed)):
            if flowerbed[i] == 0:
                left_empty = (i == 0) or (flowerbed[i - 1] == 0)
                right_empty = (i == len(flowerbed) - 1) or (flowerbed[i + 1] == 0)
                if left_empty and right_empty:
                    n -= 1
                    flowerbed[i] = 1
                    if n == 0: 
                        return True

        return False
```

## 57. Non decreasing array
- https://leetcode.com/problems/non-decreasing-array/description/

- To check the violation, we need to figure out which element to change and so we need to look back at two elements before
```py
class Solution(object):
    def checkPossibility(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        # greedy, find nums[i-1] > nums[i], increase a counter and either change nums[i-1] or nums[i]
        # if i <= 1 => change nums[i-1]
        # if i > 1 and nums[i-2] <= nums[i] => change nums[i-1]
        # else change nums[i]
        count = 0
        
        for i in range(1, len(nums)):
            if nums[i-1] > nums[i]:
                count += 1
                if count > 1: return False
                
                if i <= 1 or nums[i-2] <= nums[i]:
                    nums[i-1] = nums[i]
                else:
                    nums[i] = nums[i-1]
                    
        return True
```
