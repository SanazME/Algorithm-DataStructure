# 32 problems:
- My own interview problem
- Cron job
- Best time to buy and sell stock
- Best Time to Buy and Sell Stock II
- Jump Game
- Jump Game II
- h index
- Insert Delet GetRandom O(1)
- Product of Array except self
- Gas station
- Candy
- Zigzag conversion
- Needle in haystack
- Text Justification
- Contianer with most water
- Three Sum
- Minimum Size Subarray Sum
- Array of Doubled Pairs
- BFS
- Walls and Gates
- Number of Islands
- Open the lock
- Perfect square
- MinStack (2 approach)
- Valid Parantheses
- Daily Temp
- Reverse Polish Notation
- DFS
- Deep copy of a graph
- Target Sum
- BT in-order traversal (2 approaches)
- Decode String
- Flood fill (2)
- Path sum (2)
- water trap
- Longets consecutive sequence
- Spiral Array
- Pascal triangle
- Min size subarray sum
- Rotate Array
- Circular queue (2)
- Moving average from data stream
- Subtree of Another tree (2)
- Kth Largest Element in a Stream
- Desing a HashSet (2 large)
- Search in BST
- Insert in BST
- Delete in BST

### My own interview Q: Implement a function to compare two json templates. A template can contain either strings or nested templates. 

```py
def compareTemplates(dict1, dict2):
    output = {}
    
    # when key is not in dict2
    for key in dict1.keys():
        if key not in dict2:
            output[key] = (1, 0, None)
        else:
            val1 = dict1[key]
            val2 = dict2[key]
            
            if val1 == val2:
                continue
            else:
                if not isinstance(val1, str) and not isinstance(val2, str):
                    output[key] = compareTemplates(val1, val2)
                else:
                    output[key] = (1, 1, val2)
    # print(output, dict2)
    # when key is not in dict1
    for key in dict2.keys():
        if key not in dict1:
            output[key] = (0, 1, None)
            
    return output


# Example usage:
template2 = {
    "section1": {
        "title": "Hello",
        "content": {
            "paragraph1": "This is a test",
            "paragraph2": "Of nested templates"
        }
    },
    "section2": "Simple string"
}

template1 = {
    "section1": {
        "title": "Hello",
        "content": {
            "paragraph1": "This is a test",
            "paragraph2": "With different content"
        }
    },
    "section2": {
        "new": "Nested template instead of string"
    }
}

result = compareTemplates(template1, template2)
print(result)
```

## Cron job
- Write a data structure "Cron" to represent a single row of the file optimized for getting the next schedule time.
Write a method calculating the next time "Date getNext(Date current, Cron schedule)"
- There are 6 columns: Second, Minute, Hour, Day, Month, DayOfWeek
Each column can contain expressions:
- Star '*' stands for every/any
- Comma ',' separates values
- Dash '-' stands for range
- Slash '/' stands for period
- Combinations of the above are possible

```py
from datetime import datetime
from enum import Enum
# 0 0 0 * * *
# h min sec
# return 12 am


# 30 * * * * *
# get_next(2021-10-15 11:20:10) ======> 2021-10-15 11:20:30
# get_next(2021-10-15 11:20:30) ======> 2021-10-15 11:20:30
# get_next(2021-10-15 11:20:40) ======> 2021-10-15 11:21:30

class TimeSlice(Enum):
    SEC = 1
    MIN = 2
    HR = 3
    DAY = 4
    MONTH = 5
    DAYWEEK = 6

class Cron:
    def __init__(self, line):
        print(line)
        self.fields = line.split(' ')
        print(self.fields)
        
        self.mapEvery = {
            TimeSlice.SEC: range(60),
            TimeSlice.MIN: range(60),
            TimeSlice.HR: range(24),
            TimeSlice.DAY: range(1, 31),
            TimeSlice.MONTH: range(1,13),
            TimeSlice.DAYWEEK: range(1,8)
        }
        
        self.mapEnd = {
            TimeSlice.SEC: 60,
            TimeSlice.MIN: 60,
            TimeSlice.HR: 24,
            TimeSlice.DAY: 30,
            TimeSlice.MONTH: 12,
            TimeSlice.DAYWEEK: 7
        }
        
        self.seconds = self.parse_field(self.fields[0], TimeSlice.SEC)
        self.mins = self.parse_field(self.fields[1], TimeSlice.MIN)
        self.hours = self.parse_field(self.fields[2], TimeSlice.HR)
        self.days = self.parse_field(self.fields[3], TimeSlice.DAY)
        self.months = self.parse_field(self.fields[4], TimeSlice.MONTH)
        self.days_of_week = self.parse_field(self.fields[5], TimeSlice.DAYWEEK)
        

    def parse_field(self, field, enumVal):
        
        if field == '*':
            possibleRange = self.mapEvery[enumVal]        
            return set(possibleRange)
        
        result = set()
        for par in field.split(','):
            if '-' in par:
                start, end = map(int, par.split('-'))
                result.update(range(start, end + 1))
            elif '/' in par:
                start, step = map(int, par.split('/'))
                result.update(range(start, self.mapEnd[enumVal], step))
            else:
                result.add(int(par))
                
        return result
        
    def get_next(self, current_time):
        nextTime = current_time
        
        while True:
            if (nextTime.second in self.seconds and \
            nextTime.minute in self.mins and \
            nextTime.hour in self.hours and \
            nextTime.day in self.days and \
            nextTime.month in self.months and \
            nextTime.weekday() in self.days_of_week):
                return nextTime

            nextTime += timedelta(seconds = 1)
        
        

frequency = Cron('30 * * * * *')
nextRun = frequency.get_next(datetime(2021,10,15,11,20,10))
print(f"# get_next() ======> {nextRun.strftime('%Y-%m-%d %H:%M:%S')}")
nextRun = frequency.get_next(datetime(2021,10,15,11,20,30))
print(f"# get_next() ======> {nextRun.strftime('%Y-%m-%d %H:%M:%S')}")
nextRun = frequency.get_next(datetime(2021,10,15,11,20,40))
print(f"# get_next() ======> {nextRun.strftime('%Y-%m-%d %H:%M:%S')}")  
```


### Best Time to Buy and Sell Stock
- https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/?envType=study-plan-v2&envId=top-interview-150
- Based on the fact that we have to sell after we buy and we are trying to maximize profit, we can iterate through the prices and only need to consider two things:
1.  Is this price cheaper than any other price I've seen before?
2.  If I subtract current price by the cheapest price I've found, does this yield a greater profit than what I've seen so far?

A fun thing to note is if #1 is true, then #2 cannot be true as well so there isn't a need to check
```py
def maxProfit(self, prices: 'List[int]') -> 'int':
        if len(prices) <= 1:
            return 0

        minPrice = float('Inf')
        maxProfit = 0

        for i in range(len(prices)):
            if prices[i] < minPrice:
                minPrice = prices[i]
                # maxProfit = 

            if prices[i] - minPrice > maxProfit:
                maxProfit = prices[i] - minPrice

        return maxProfit
```

### Best Time to Buy and Sell Stock II
- https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/description/
```py
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        profit = 0
        if len(prices) == 1:
            return profit

        profit = 0
        for i in range(1, len(prices)):
            diff = prices[i] - prices[i-1]
            if diff > 0:
                profit += diff
```
 


### Jump Game
- https://leetcode.com/problems/jump-game/description/
- **Explaination 1:** Starting from one element before the last one in end of the array, the goal is to reach the last element meaning doing a jump equal equal at least the index of the last element `len(nums) - 1`. If we can do it, then we update the goal to be the element before the last elements and we keep checking.
```py
def canJump(nums):
    if len(nums) <= 1:
         return True

     n = len(nums)
     goal = n - 1

     for i in range(n - 2, -1, -1):
         if i + nums[i] >= goal:
             goal = i

     return True if goal == 0 else False
```

- **Explaination 2:** Imagine you have a car, and you have some distance to travel (the length of the array). This car has some amount of gasoline, and as long as it has gasoline, it can keep traveling on this road (the array). Every time we move up one element in the array, we subtract one unit of gasoline. However, every time we find an amount of gasoline that is greater than our current amount, we "gas up" our car by replacing our current amount of gasoline with this new amount. We keep repeating this process until we either run out of gasoline (and return false), or we reach the end with just enough gasoline (or more to spare), in which case we return true.
Note: We can let our gas tank get to zero as long as we are able to gas up at that immediate location (element in the array) that our car is currently at.
```py
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        if len(nums) <= 1:
            return True

        gas = 0
        for n in nums:
            if gas < 0:
                return False

            elif n > gas:
                gas = n
            gas -= 1

        return True
```
**Note: we can also use BFS approach which is less efficient than greedy jump- explained in Jump Game II**

### Jump Game II
- https://leetcode.com/problems/jump-game-ii/description
- We can think of this as a level-by-level progression through the array. At each level, we want to make the jump that takes us the furthest. We increment our jump count each time we need to make a new jump. So, we calculate the maximum index we can reach from the current index. If our pointer i reaches the last index that can be reached with current number of jumps then we have to make a jumps.
So, we increase the count.
- Edge cases:
  - List of one element: 0 jump
  - first element in the list is 0: impossible
  - first element in the list reaches the end of the list: 1 jump 
```py
class Solution:
    def jump(self, nums):
        if len(nums) <= 1:
            return 0

        if nums[0] >= len(nums) - 1:
            return 1
        
        # Initialize reach (maximum reachable index), count (number of jumps), and last (rightmost index reached)
        reach, count, last = 0, 0, 0
        
        # Loop through the array excluding the last element
        for i in range(len(nums)-1):    
            # Update reach to the maximum between reach and i + nums[i]
            reach = max(reach, i + nums[i])

            if reach >= len(nums) - 1:
                count +=1
                break
        
            # If i has reached the last index that can be reached with the current number of jumps
            if i == last:
                # Update last to the new maximum reachable index
                last = reach
                # Increment the number of jumps made so far
                count += 1
        
        # Return the minimum number of jumps required
        return count
```

- **BFS appoach:** less efficient. We treat each index as a node in the tree and the value at that Each index in the array can be thought of as a node in a graph. The possible jumps from each index represent the edges to other nodes.
```py
from collections import deque

class Solution:
    def jump(self, nums: List[int]) -> int:
        if len(nums) <= 1:
            return 0

        if nums[0] >= len(nums) - 1:
            return 1

        queue = deque([0])
        depth = 0
        visited = set()

        while queue:
            size = len(queue)
            depth += 1

            for _ in range(size):
                i = queue.popleft()
                if i in visited: continue
                visited.add(i)
                if i + nums[i] >= len(nums) - 1:
                    return depth

                for j in range(1, nums[i] + 1):
                    queue.append(i + j)

        return depth
```
### H-Index
- https://leetcode.com/problems/h-index/description/

**approach 1:** sort the array and try to find max of min between the value in the array and how far it is from the end of array. First think of edge cases and check if the solution can cover those as well:
- one elements arrays: if [0] vs [4]
- two or more element array with all zeros
```py
    def hIndex(self, citations: List[int]) -> int:
        if len(citations) == 0:
            return 0

        # if len(citations) == 1:
        #     if citations[0] == 0:
        #         return 0
        #     else:
        #         return 1

    
        citations.sort()

        maxSoFar = -float('Inf')
        for i in range(len(citations) - 1, -1, -1):
            maxSoFar = max(maxSoFar, min(citations[i], len(citations) - i))

        return maxSoFar
```

**approach 2**: another way is createa  tmp array / bucket to hold the count of papers with frequency citations equal to the index of that bucket. Then iterate backward through that tmp array and accumulate the total number of citations up to each index i. If the total is larger or equal to that index i then that index is h index:
```py
class Solution:
    def hIndex(self, citations: List[int]) -> int:
        n = len(citations)
        temp = [0 for _ in range(n + 1)]

        for i,v in enumerate(citations):
            if v > n :
                temp[n] += 1
            else:
                temp[v] += 1
        
        total = 0
        for i in range(n, -1, -1):
            total += temp[i]
            if total >= i:
                return i
```

### Insert Delete GetRandom O(1)
- https://leetcode.com/problems/insert-delete-getrandom-o1/
- In python, creating a simple api for a set() would be a perfect solution if not for the third operation, getRandom(). We know that we can retrieve an item from a set, and not know what that item will be, but that would not be actually random. (This is due to the way python implements sets. In python3, when using integers, elements are popped from the set in the order they appear in the underlying
hashtable. Hence, not actually random.)

A set is implemented essentially the same as a dict in python, so the time complexity of add / delete is on average O(1). When it comes to the random function, however, we run into the problem of needing to convert the data into a python list in order to return a random element. That conversion will add a significant overhead to getRandom, thus slowing the whole thing down.

Instead of having to do that type conversion (set to list) we can take an approach that involves maintaining both a list and a dictionary side by side. 
```py
class RandomizedSet:

    def __init__(self):
        self.data_map = {} # dictionary, aka map, aka hashtable, aka hashmap
        self.data = [] # list aka array

    def insert(self, val: int) -> bool:

        # the problem indicates we need to return False if the item 
        # is already in the RandomizedSet---checking if it's in the
        # dictionary is on average O(1) where as
        # checking the array is on average O(n)
        if val in self.data_map:
            return False

        # add the element to the dictionary. Setting the value as the 
        # length of the list will accurately point to the index of the 
        # new element. (len(some_list) is equal to the index of the last item +1)
        self.data_map[val] = len(self.data)

        # add to the list
        self.data.append(val)
        
        return True

    def remove(self, val: int) -> bool:

        # again, if the item is not in the data_map, return False. 
        # we check the dictionary instead of the list due to lookup complexity
        if not val in self.data_map:
            return False

        # essentially, we're going to move the last element in the list 
        # into the location of the element we want to remove. 
        # this is a significantly more efficient operation than the obvious 
        # solution of removing the item and shifting the values of every item 
        # in the dicitionary to match their new position in the list
        last_elem_in_list = self.data[-1]
        index_of_elem_to_remove = self.data_map[val]

        self.data_map[last_elem_in_list] = index_of_elem_to_remove
        self.data[index_of_elem_to_remove] = last_elem_in_list

        # change the last element in the list to now be the value of the element 
        # we want to remove
        self.data[-1] = val

        # remove the last element in the list
        self.data.pop()

        # remove the element to be removed from the dictionary
        self.data_map.pop(val)
        return True

    def getRandom(self) -> int:
        # if running outside of leetcode, you need to `import random`.
        # random.choice will randomly select an element from the list of data.
        return random.choice(self.data)
```

### Product of Array except self
- https://leetcode.com/problems/product-of-array-except-self
- calculate suffix and prefix array for each element
**with extra space**
```py
def productExceptSelf(nums):
    if len(nums) <= 1:
        return None
    
    prefix, suffix = [1] * len(nums),[1] * len(nums)
    output = [0] * len(nums)
    
    for i in range(len(nums) - 2, -1, -1):
        suffix[i] = nums[i + 1] * suffix[i + 1]
        
    for i in range(1, len(nums)):
        prefix[i] = nums[i - 1] * prefix[i - 1]
    
    for i in range(len(nums)):
        output[i] = prefix[i] * suffix[i]
        
    return output
```
**without extra space**
```py
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        if len(nums) <= 1:
            return None
        
        prefix, suffix = 1, 1
        output = [0] * len(nums)
        
        for i in range(len(nums) - 1, -1, -1):
            output[i] = suffix
            suffix = nums[i] * suffix
            
        for i in range(len(nums)):
            output[i] *= prefix
            prefix *= nums[i]
        
    
        return output
```

### Gas station
- https://leetcode.com/problems/gas-station/description/

```py
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        if sum(gas) - sum(cost) < 0:
            return -1

        gas_tank, startIdx = 0, 0

        for i in range(len(gas)):
            gas_tank += gas[i] - cost[i]
            if gas_tank < 0:
                startIdx = i + 1
                gas_tank = 0

        return startIdx
```
### Candy
- https://leetcode.com/problems/candy/description/
- We can solve it in Time O(n) and Space O(n) looping over the array twice - we can even optimize it more by looping once with space O(1):
```py
# Time: O(n) and Space O(n) - two passes
        if len(ratings) <= 1:
            return len(ratings)

        candy = [1] * len(ratings)

        for i in range(1, len(ratings)):
            if ratings[i] > ratings[i - 1]:
                candy[i] = candy[i - 1] + 1

        for i in range(len(ratings) - 2, -1, -1):
            if ratings[i] > ratings[i + 1]:
                candy[i] = max(candy[i], candy[i + 1] + 1)

        return sum(candy)
```
- One-Pass Greedy Algorithm: Up-Down-Peak Method
Why Up, Down, and Peak?
The essence of the one-pass greedy algorithm lies in these three variables: Up, Down, and Peak. They serve as counters for the following:
* **Up**: Counts how many children have increasing ratings from the last child. This helps us determine how many candies we need for a child with a higher rating than the previous child.
* **Down**: Counts how many children have decreasing ratings from the last child. This helps us determine how many candies we need for a child with a lower rating than the previous child.
* **Peak**: Keeps track of the last highest point in an increasing sequence. When we have a decreasing sequence after the peak, we can refer back to the Peak to adjust the number of candies if needed.
How Does it Work?
1. Initialize Your Counters
   * Start with ret = 1 because each child must have at least one candy. Initialize up, down, and peak to 0.
2. Loop Through Ratings
   * For each pair of adjacent children, compare their ratings. Here are the scenarios:
      * If the rating is increasing: Update **up** and **peak** by incrementing them by 1. Set **down** to 0. Add up + 1 to **ret** because the current child must have one more candy than the previous child.
      * If the rating is the same: Reset **up**, **down**, and **peak** to 0, because neither an increasing nor a decreasing trend is maintained. Add 1 to **ret** because the current child must have at least one candy.
      * If the rating is decreasing: Update **down** by incrementing it by 1. Reset **up** to 0. Add **down** to **ret**. Additionally, **if peak is greater than or equal to down, decrement ret by 1. This is because the peak child can share the same number of candies as one of the children in the decreasing sequence, which allows us to reduce the total number of candies.**

This subtraction is only done when peak >= down because that's when we know we can "borrow" a candy from the peak without violating the rules. Once down > peak, we can no longer borrow from the peak, so we stop subtracting. This subtraction is safe because as long as peak >= down, we know that the peak still has more candies than the current decreasing element, even after subtracting 1.
```py
# Time: O(n) and Space O(1) - one pass: runtime optimization
        if len(ratings) <= 0:
            return len(ratings)

        up, down, peak = 0, 0, 0
        total = 1

        for i in range(1, len(ratings)):
            if ratings[i] > ratings[i - 1]:
                up += 1
                down = 0
                peak = up
                total += up + 1

            elif ratings[i] < ratings[i - 1]:
                down += 1
                up = 0
                if peak >= down:
                    total += down + 1 - 1
                else:
                    total += down + 1
                
            else:
                up = down = peak = 0
                total += 1

        return total
```
### zigzag conversion
- https://leetcode.com/problems/zigzag-conversion/description
```py
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows <= 1:
            return s
        
        order = {}
        for i in range(1, numRows + 1):
            order[i] = ''

        print(order)

        counter = 1
        direction = 'up'

        for char in s:
            # correct direction and reset counter
            
            if counter > numRows:
                counter -= 2
                direction = 'down'
            elif counter <= 0:
                counter += 2
                direction = 'up'
            
            # print(char, counter, direction)
            # populate order dictionary
            if direction == 'up':
                order[counter] += char
                counter += 1

            elif direction == 'down':
                order[counter] += char
                counter -= 1

        # print(order)
        result = ''
        for i in range(1, numRows + 1):
            result += order[i]

        return result
```

### Needle in haystack
- https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string/
```py
def strStr(self, haystack: str, needle: str) -> int:
        if len(needle) == 0:
            return 0

        if len(needle) > len(haystack):
            return -1
        
        pointer = 0
        
        for i in range(len(haystack)):
            
            if i + len(needle) > len(haystack):
                break
            
            for j in range(len(needle)):
                if needle[j] != haystack[i + j]:
                    break
                    
                if j == len(needle) - 1:
                    return i
                
        return -1
```

### Text Justification
- https://leetcode.com/problems/text-justification/description/
- Adding extra space from left to right meaning a round robin approach: we first add required space evently to after each word and then for extra spaces, we add them from left to right again.
```py
class Solution:
    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:

        result = []
        curr_line = []
        curr_line_length = 0

        def justify_line(line, isLastLine = False):
            if isLastLine or len(line) == 1:
                return ' '.join(line).ljust(maxWidth)

            total_spaces = maxWidth - sum(len(word) for word in line)
            gaps = len(line) - 1
            if gaps > 0:
                spaces_per_gap = total_spaces // gaps
                extra_spaces = total_spaces % gaps

            justified = []
            for i, word in enumerate(line):
                justified.append(word)
                if i < gaps:
                    spaces = spaces_per_gap + (1 if i < extra_spaces else 0)
                    justified.append(' ' * spaces)
                elif gaps == 0:
                    justified.append(' ' * spaces)

            return ''.join(justified)


        for word in words:
            if curr_line_length + len(word) + len(curr_line) <= maxWidth:
                curr_line.append(word)
                curr_line_length += len(word)
            else:
                result.append(justify_line(curr_line))
                curr_line = [word]
                curr_line_length = len(word)

        if curr_line:
            result.append(justify_line(curr_line, True))
        
        return result
```

### Container with most water
- https://leetcode.com/problems/container-with-most-water/?envType=study-plan-v2&envId=top-interview-150

- use two pointers starting from start and end and always move the shorter pointer.
```py
class Solution:
    def maxArea(self, height: List[int]) -> int:
        if len(height) <= 1:
            return 0

        if len(height) == 2:
            return min(height)


        left, right = 0, len(height) - 1
        volume = 0

        while left < right:
            volume = max(volume, min(height[left], height[right]) * (right - left))
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1


        return volume
```

### Three Sum
- https://leetcode.com/problems/3sum
```py
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        result = set()
        dups = set()
        seen = {}

        if len(nums) < 3:
            return result

        for i, num1 in enumerate(nums):
            if num1 not in dups:
                dups.add(num1)
                for j, num2 in enumerate(nums[i+1:]):
                    complement = -num1 - num2
                    if complement in seen and seen[complement] == i:
                        result.add(tuple(sorted((num1, num2, complement))))
                    seen[num2] = i

        print(result)
        return result
```
- to make sure the uniqueness, we add the element from the main loop to `dups` set because if we already process for that element. we know the answers and we don't need to do it again later but also we have `seen` dictionary which add any seen element (seconds loop) as key and its value to ele from main loop. This way, we know when we visit that element (during which element from the main loop). Like [1,0,-1, 1, 2, -3], when j == -1, we already have 0 in seen associated with i = 1 but for the second 1 not yet.

### Minimum Size Subarray Sum
- https://leetcode.com/problems/minimum-size-subarray-sum/description
- since we don't want to order the array `N logN`, we start adding numbers and once the sum becomes bigger than target, we keep remove elements from left to right till that sum becomes smaller. This way we can find the smallest subarray that its sum is at least equal to target.
- The idea behind it is to maintain two pointers: start and end, moving them in a smart way to avoid examining all possible values 0<=end<=n-1 and 0<=start<=end (to avoid brute force).
What it does is:
1. Incremeting the end pointer while the sum of current subarray (defined by current values of start and end) is smaller than the target.
2. Once we satisfy our condition (the sum of current subarray >= target) we keep incrementing the start pointer until we violate it (until `sum(array[start:end+1]) < target`).
3. Once we violate the condition we keep incrementing the end pointer until the condition is satisfied again and so on.

```py
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        maxVal = max(nums)
        if maxVal >= target:
            return 1

        if len(nums) == 0:
            return 0

        left = 0
        minCount = float('Inf')
        sumSoFar = 0
        
        
        for i in range(len(nums)):
            sumSoFar += nums[i]
            
            while sumSoFar >= target:
                minCount = min(minCount, i - left + 1)
                sumSoFar -= nums[left]
                left += 1
                
        if minCount == float('Inf'):
            return 0
        else:
            return minCount
```

### Subarrays with k Different Integers
- https://leetcode.com/problems/subarrays-with-k-different-integers/description/
- The key insight is that we can transform this problem into a sliding window problem. Instead of directly counting subarrays with exactly K different integers, we can use the following trick:
**(number of subarrays with at most K different integers) - (number of subarrays with at most K-1 different integers)**

This works because any subarray with exactly K different integers is included in the count of subarrays with at most K different integers, but not in the count of subarrays with at most K-1 different integers.

So, we need to create a function that counts subarrays with at most K different integers, and then use it twice.
For the "at most K" function, we'll use a sliding window approach with two pointers (left and right) and a Counter to keep track of the unique integers in the current window.

- as for how to count number of subarrays when we slide the window: The key insight here is that every time we add a new element (move the right pointer), we're not just adding one new subarray, but potentially many. Here's how it works:

1. At each step, `right - left + 1` represents the length of the current window.
2. This length also equals the number of subarrays that end at the current right pointer and still satisfy our "at most K distinct integers" condition.

The key points to understand are:

1. Every time we add a new element, we're creating new subarrays that end with this element.
2. The number of these new subarrays is equal to the current window length `(right - left + 1)`.
3. This works because for each existing subarray that satisfied our condition, adding the new element creates a new valid subarray.

Consider the array [1, 2, 1, 2] with k = 2.
Step 1: [1]

Window: [1]
right - left + 1 = 0 - 0 + 1 = 1
New subarrays: [1]
result += 1

Step 2: [1, 2]

Window: [1, 2]
right - left + 1 = 1 - 0 + 1 = 2
New subarrays: [2], [1, 2]
result += 2

Step 3: [1, 2, 1]

Window: [1, 2, 1]
right - left + 1 = 2 - 0 + 1 = 3
New subarrays: [1], [2, 1], [1, 2, 1]
result += 3

Step 4: [1, 2, 1, 2]

Window: [1, 2, 1, 2]
right - left + 1 = 3 - 0 + 1 = 4
New subarrays: [2], [1, 2], [2, 1, 2], [1, 2, 1, 2]
result += 4

Final result: 1 + 2 + 3 + 4 = 10

```py
from collections import Counter

class Solution:
    def subarraysWithKDistinct(self, nums: List[int], k: int) -> int:
        return self.atMostK(nums, k) - self.atMostK(nums, k - 1)
    
    def atMostK(self, nums: List[int], k: int) -> int:
        count = Counter()
        left = 0
        result = 0
        
        for right in range(len(nums)):
            if count[nums[right]] == 0:
                k -= 1
            count[nums[right]] += 1
            
            while k < 0:
                count[nums[left]] -= 1
                if count[nums[left]] == 0:
                    k += 1
                left += 1
            
            result += right - left + 1
        
        return result
```

### Max Consecutive Ones III
- https://leetcode.com/problems/max-consecutive-ones-iii/description/
```py
def longestOnes(self, nums: List[int], k: int) -> int:
        if len(nums) == 0:
            return None

        left, result = 0, 0

        for right in range(len(nums)):
            if nums[right] == 0:
                k -= 1

            while k < 0:
                if nums[left] == 0:
                    k += 1
                left += 1

            result = max(result, right - left + 1)
        
        return result
```
- similar to prev question, sliding windows and the size of the window is dynamic and can get as large as long as we can still flip zeros in the subarray.

###  Shortest Subarray with Sum at Least K
- https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/description/
- the difference of this from problem with positive inegers only is the persence of negative numbers. Sliding window alone would work for case of positive integer array. The idea behind it is to maintain two pointers: start and end, moving them in a smart way to avoid examining all possible values `0<=end<=n-1` and `0<=start<=end` (to avoid brute force).
What it does is:

1. Incremeting the `end` pointer while the sum of current subarray (defined by current values of `start` and `end`) is smaller than the target.
2. Once we satisfy our condition (the sum of current `subarray >= target`) we keep incrementing the start pointer until we violate it (until `sum(array[start:end+1]) < target`).
3. Once we violate the condition we keep incrementing the end pointer until the condition is satisfied again and so on.
 
The reason why we stop incrementing start when we violate the condition is that we are sure we will not satisfy it again if we keep incrementing start. In other words, if the sum of the current subarray `start -> end` is smaller than the target then the sum of `start+1 -> end` is neccessarily smaller than the target. (positive values). The problem with this solution is that it doesn't work if we have negative values, this is because of the sentence above **Once we "violate" the condition we stop incrementing start**.

**Problem of the sliding windows with negative values**
Now, let's take an example with negative values `nums = [3, -2, 5]` and `target=4`. Initially `start=0`, we keep moving the end pointer until we satisfy the condition, here we will have `start=0 and end=2`. Now we are going to move the start pointer `start=1`. The sum of the current subarray is `-2+5=3 < 4` so we violate the condition. However if we just move the `start` pointer another time `start=2` we will find `5 >= 4` and we are satisfying the condition. And this is not what the Sliding window assumes.

We need to use deque and it's a modiifcation of sliding window solution.

**What does the Deque store :**
The deque stores the possible values of the start pointer. Unlike the sliding window, values of the `start` variable will not necessarily be contiguous.

**Why is it increasing :**
So that when we move the start pointer and we violate the condition, we are sure we will violate it if we keep taking the other values from the Deque. In other words, if the sum of the subarray from `start=first` value in the deque to end is smaller than `target`, then the sum of the subarray from `start=second` value in the deque to end is necessarily smaller than target.
So because the Deque is increasing `(B[d[0]] <= B[d[1]])`, we have `B[i] - B[d[0]] >= B[i] - B[d[1]]`, which means the sum of the subarray starting from `d[0]` is greater than the sum of the sub array starting from `d[1]`.

**Why do we have a prefix array and not just the initial array like in sliding window :**
Because in the sliding window when we move `start` (typically when we increment it) we can just substract `nums[start-1]` from the current sum and we get the sum of the new subarray. Here the value of the `start` is jumping and **one way to compute the sum of the current subarray in a constant time is to have the prefix array**.

**Why using Deque and not simply an array :**
We can use an array, however we will find ourselves doing only three operations:
1. `remove_front` : when we satisfy our condition and we want to move the start pointer
2. `append_back` : for any index that may be a future start pointer
3. `remove_back` : When we are no longer satisfying the **increasing order of the array**
Deque enables doing these 3 operations in a constant time.

```py
from collections import deque

class Solution:
    def shortestSubarray(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        result = n + 1  # Initialize with a value larger than possible subarrays
        d = deque()
        
        for i in range(n + 1):
            while d and prefix_sum[i] - prefix_sum[d[0]] >= k:
                result = min(result, i - d.popleft())
            
            while d and prefix_sum[i] <= prefix_sum[d[-1]]:
                d.pop()
            
            d.append(i)
        
        return result if result <= n else -1
```

### Longest Substring Without Repeating Characters
- Similer to previous solutions
```py
def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        if len(s) <= 1:
            return len(s)
        
        left = 0
        count = Counter()
        result = 0
        
        for right in range(len(s)):
            count[s[right]] += 1
            
            while count[s[right]] > 1:
                count[s[left]] -= 1
                left += 1
            
            result = max(result, right - left + 1)
            
                
        return result
```
### Fruit Into Baskets
- https://leetcode.com/problems/fruit-into-baskets/description/

- This is also the same problem as sliding window for AtMost 2 different types of fruits, we want to return the longest subarray:
```py
class Solution:
    def totalFruit(self, fruits: List[int]) -> int:
        return self.atMost(fruits, 2)


    def atMost(self, fruits, k):
        result = 0
        left = 0
        count = Counter()

        for right in range(len(fruits)):
            if count[fruits[right]] == 0:
                k -= 1

            count[fruits[right]] += 1
            while k < 0:
                count[fruits[left]] -= 1
                if count[fruits[left]] == 0:
                    k += 1
                left += 1

            result = max(result, right - left + 1)

        return result
```

## Count Number of Nice Subarrays
- https://leetcode.com/problems/count-number-of-nice-subarrays/description/

1. we can use sliding window and just count the number of odd numbers that we encountered so far.
2. we calculate all valud subarrays that have **at most** K odd numbers using the lenght of the sliding window.
3. Once the count goes beyond k we can start increasing `left` pointer.

```py
 def numberOfSubarrays(self, nums: List[int], k: int) -> int:
        def atMost(k):
            count = res = left = 0
            for right in range(len(nums)):
                count += nums[right] % 2
                while count > k:
                    count -= nums[left] % 2
                    left += 1
                res += right - left + 1
            return res
        
        return atMost(k) - atMost(k-1)
```

### Binary Subarrays With Sum
- https://leetcode.com/problems/binary-subarrays-with-sum/description/

- be aware of edge case when goal is already 0!
```py
    def numSubarraysWithSum(self, nums: List[int], goal: int) -> int:
        if goal == 0:
            return self.atMost(nums, goal)

        return self.atMost(nums, goal) - self.atMost(nums, goal - 1)

    
    def atMost(self, nums, goal):
        left = 0
        result = 0
        goalSoFar = 0

        for right in range(len(nums)):
            goalSoFar += nums[right]

            while goalSoFar > goal:
                goalSoFar -= nums[left]
                left += 1
                
            result += right - left + 1

        return result
```
### Substring with Concatenation of All Words
- https://leetcode.com/problems/substring-with-concatenation-of-all-words/description/?envType=study-plan-v2&envId=top-interview-150
1. all words in `words` array have the same lenght: `word_lenght`
2. the total length of all words concatenated will be `total_length=word_length * len(words)`
3. we need to check every possible substring of length `total_length` in string `s`.
4. we can defined a sliding window of size `total_length` and slide it one element at a time.
5. For each iteration on that sliding window, we split the current substring of `total_lenght` into `word_length` chunks
6. we count frequency of those words and compare the hash map to the hash map of our words list
7. if those two are equal, we've found an answer and add it to the collection of our answers

**After main solution we can do two optimization**

**Time complexity**
- `n` length of the string s
- `m` number of words
- `k` length of each word
1. Outer loop: `O(n)`
2. Inner loop: `O(m * k)`
`---> O(n * m * k)`

**Space complexity**
1. `countShould`: `O(m)`
2. `chunks` : `O(k * m)`
3. `countIs` : `O(m)`
4. `result` : `O(n)` the worst case where every position is a valid starting index
`---> O(k * m)` the dominant factor

```py
from collections import Counter
class Solution:
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        if len(words) == 0 or len(s) == 0:
            return []

        countShould = Counter(words)
        word_length = len(words[0])
        total_length = word_length * len(words)

        if len(s) < total_length:
            return []

        left = 0
        result = []

        while left + total_length <= len(s):
            chunks = [s[i:i + word_length] for i in range(left, left + total_length, word_length)]
            countIs = Counter(chunks)
            if countIs == countShould:
                result.append(left)
                
            left += 1


        return result
```
**Optimization: Use Rolling hash to avoid creating the whole chunks list everytime, we just change one element**
- Rolling hash for substrings: https://cp-algorithms.com/string/string-hashing.html
- https://github.com/SanazME/Algorithm-DataStructure/blob/master/README.md#rolling-hash-and-string-hashing


```py
from typing import List
from collections import Counter

class Solution:
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        if not s or not words:
            return []

        word_length = len(words[0])
        word_count = len(words)
        total_length = word_length * word_count

        if len(s) < total_length:
            return []

        p = 31
        m = 10**9 + 7

        def compute_rolling_hash(string: str) -> List[int]:
            h = [0] * (len(string) + 1)
            p_pow = 1
            for i in range(1, len(string) + 1):
                h[i] = (h[i-1] + (ord(string[i-1]) - ord('a') + 1) * p_pow) % m
                p_pow = (p_pow * p) % m
            return h

        def get_substring_hash(h: List[int], start: int, end: int) -> int:
            return (h[end] - h[start]) * pow(p, -start, m) % m

        # Compute hashes for all words
        word_hashes = {}
        for word in words:
            word_hash = compute_rolling_hash(word)[-1]
            word_hashes[word_hash] = word_hashes.get(word_hash, []) + [word]

        # Compute rolling hash for s
        s_hash = compute_rolling_hash(s)

        result = []
        word_counter = Counter(words)

        for i in range(word_length):
            left = i
            seen = Counter()

            for j in range(left, len(s) - word_length + 1, word_length):
                curr_hash = get_substring_hash(s_hash, j, j + word_length)
                
                if curr_hash in word_hashes:
                    word = word_hashes[curr_hash][0]  # Take the first matching word
                    seen[word] += 1

                    while seen[word] > word_counter[word]:
                        seen[word_hashes[get_substring_hash(s_hash, left, left + word_length)][0]] -= 1
                        left += word_length

                    if j + word_length - left == total_length:
                        result.append(left)
                else:
                    seen.clear()
                    left = j + word_length

        return result

# Example usage
sol = Solution()
s = "barfoothefoobarman"
words = ["foo","bar"]
print(sol.findSubstring(s, words))  # Expected output: [0, 9]
```

### Minimum Window Substring
- https://leetcode.com/problems/minimum-window-substring/description


- with sliding window, we increase the window size till we have all elements with correct frequency in the window, we then start shrinking the window from left as far as we still have all elements with correct frequency and then we save the smallest substring we found so far.

```py
import collections
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if s is None or t is None:
            return ""
    
        if len(t) > len(s):
            return ""
        
        dict_t = collections.Counter(t)
        required = len(dict_t)
        formed = 0
        window_counts = {}
        result = (float('Inf'), None, None)
        
        
        left, right = 0, 0
        
        while right < len(s):
            char = s[right]
            
            window_counts[char] = window_counts.get(char, 0) + 1
            
            if char in dict_t and window_counts[char] == dict_t[char]:
                formed += 1
                
            
            while formed == required and left <= right:
                char = s[left]
                
                if right - left + 1 < result[0]:
                    result = (right - left + 1, left, right)
                
                window_counts[char] -= 1
                
                if char in dict_t and window_counts[char] < dict_t[char]:
                    formed -= 1
                
                
                left += 1
            
            right += 1
            
        return "" if result[0] == float('inf') else s[result[1]: result[2] + 1]
```    
            
        





### Array of Doubled Pairs
- https://leetcode.com/problems/array-of-doubled-pairs/
- for each element in array, x, we need to find whether `2*x or x/2` exist. However, if we sort the array based on their abs value, then we need to only check for the existence of `2*x` because, x is the least value and so `x/2` can not exist.
- We might have double or more than occurance of numbers so we want to keep count of values we visited and remove them from the count so we don't use the same value twice. FOr that we need a hashmap of values and their counts.
- The time complexity is `O(NlogN)`. `N` is for creating a hashmap of values and their counts and even though we iterate on the sorted array, we only visit half of it each time because we remove the two visited ones. So if the lenght was 8, next time 6 .... Every time, we're going to look at half of the values and map the rest with its occurances


# Breadth First Search (BFS)

- One common application of Breadth-first Search (BFS) is to find the shortest path from the root node to the target node. BFS of a graph is similar to BFS of a tree. The only catch is, unlike tree, graphs may contain cycles. so we may come to th same node.To avoid processing a node more than once, we use a boolean visited array. 
- https://www.programiz.com/dsa/graph-bfs
- If a node X is added to the queue in the kth round, the length of the shortest path between the root node and X is exactly k. That is to say, you are already in the shortest path the first time you find the target node.
- The time complexity in a graph is O(V+E), where V: number of vertices and E: number of edges.

```py
# BFS algorithm in Python
import collections

def bfs(graph, root):
    queue = collections.deque([root])
    visited = set()
    depth = -1
    
    while queue:
        size = len(queue)
        depth += 1
        
        for _ in range(size):
            cur = queue.popleft()
            if cur in visited: continue
            visited.add(cur)
            queue.extend(nextLayer(node))
        
def nextLayer(node):
    listNodes = []
    // add all successors of node to listNodes
    return listNodes
    
            
if __name__ == '__main__':
    graph = {0: [1, 2], 1: [2], 2: [3], 3: [1, 2]}
    print("Following is Breadth First Traversal: ")
    bfs(graph, 0)
```

## Walls and Gates
- https://leetcode.com/problems/walls-and-gates/
- Instead of searching from an empty room to the gates, we can BFS form all gates at the same time. Since BFS guarantees that we search all rooms of distance `d` before searching rooms of distance `d+1`, the distance to an empty room must be the shortest.

```py
import collections

class Solution:
    def wallsAndGates(self, rooms: List[List[int]]) -> None:
        """
        Do not return anything, modify rooms in-place instead.
        """
        if not rooms:
            return
        
        rows = len(rooms)
        cols = len(rooms[0])
        
        for i in range(rows):
            for j in range(cols):
                if rooms[i][j] == 0:
                    queue = collections.deque([])
                    visited = set()
                    queue.append((i+1, j, 1))
                    queue.append((i-1, j, 1))
                    queue.append((i, j+1, 1))
                    queue.append((i, j-1, 1))
                    
                    while queue:
                        x, y, val = queue.popleft()
                        if x < 0 or x >= rows or y < 0 or y >= cols or (x,y) in visited or rooms[x][y] in [-1, 0]:
                            continue
                            
                        visited.add((x, y))
                        rooms[x][y] = min(rooms[x][y], val)
                        queue.append((x + 1, y, val + 1))
                        queue.append((x - 1, y, val + 1))
                        queue.append((x, y + 1, val + 1))
                        queue.append((x, y - 1, val + 1))
```

## Number of Islands:

- BFS, we search for each island:
```py
def numIslands(grid):
    if not grid:
        return
    
    visited = set()
    queue = collections.deque([])
    rows, cols = len(grid), len(grid[0])
    count = 0
    
    
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '1' and (i, j) not in visited:
                count += 1
                queue.append((i,j))
                
                while queue:
                    x, y = queue.popleft()
                    if x < 0 or x >= rows or y < 0 or y >= cols or (x,y) in visited or grid[x][y] != '1':
                        continue
                        
                    visited.add((x,y))
                    queue.append((x + 1, y))
                    queue.append((x - 1, y))
                    queue.append((x, y + 1))
                    queue.append((x, y - 1))
                
                
    return count
```
- https://github.com/SanazME/Algorithm-DataStructure/blob/master/AmazonQuestions.md#5-number-of-islands


## Open the lock
- https://leetcode.com/problems/open-the-lock/
- We can think of the problem as a shortest path on a graph: there are 10,000 nodes (strings `0000` to '9999`) and there is an edge between two nodes if they differ in one digit, and if both nodes are not in `deadends`.

```py
import collections

class Solution:
    def openLock(self, deadends: List[str], target: str) -> int:
        
        queue = collections.deque([('0000', 0)])
        visited = set()
        dead = set(deadends)
        
        def getNeighbors(node):
            output = []
            for i, char in enumerate(node):
                digit = int(char)
                output.append(node[:i] + str((digit + 1) % 10) + node[i+1:])
                output.append(node[:i] + str((digit - 1) % 10) + node[i+1:])
            return output
        
        while queue:
            curr, depth = queue.popleft()
            
            if curr == target:
                return depth
            
            if curr in visited or curr in dead:
                continue
                
            visited.add(curr)
            neighbours = getNeighbors(curr)
            
            for neighbor in neighbours:
                queue.append((neighbor, depth + 1))
                
        return -1
 ```

**Time Complexity : `O(A^N * N^2 + D)`**:
- `N` is the number of dials (4 in our case)
- `A` is the number of alphabets (10 in our case -> 0 to 9)
- `D` is the size of deadends

There are 10*10*10*10 possible combinations --> `A^N`
for each combination, we are looping 4 times (which is `N`) and in each iteration, there are substring operations (`O(N)`):
`A^N * N^2 + D`

**Space complexity: `O(A^N + D)`**

## Perfect Squares
- https://leetcode.com/problems/perfect-squares/
- We consider an N-ary tree where each node represents a remainder of the number `n` substracting a combination of square numbers. We can use BFS to find the minimal number of square numbers that add up to our original number. To avoid doing the same calculations for the same value of remainder, we use `visited` set.

```py
def numSquares(self, n: int) -> int:
        
    squareNums = [i*i for i in range(1, int(n**0.5) + 1)]
    print(squareNums)

    queue = collections.deque([(n, 0)])
    visited = set()

    while queue:
        curr, depth = queue.popleft()
        if curr == 0:
            return depth

        if curr in visited:
            continue

        visited.add(curr)

        for perfectNum in squareNums:
            if curr < perfectNum:
                break
            else:
                queue.append((curr - perfectNum, depth + 1))
```

## Min Stack
- https://leetcode.com/problems/min-stack/
- Option 1: use a linked list and each node has a min val, we push and pop the head.
```py
class Node:
    def __init__(self, val = None):
        self.val = val
        self.next = None
        self.min = float('Inf')

class MinStack:
    def __init__(self):
        self.head = Node()
        
    def push(self, val):
        curr = self.head
        node = Node(val)
        node.min = min(curr.min, val)
        node.next = curr
        self.head = node
        
    def pop(self):
        curr = self.head
        nextNode = self.head.next
        curr.next = None
        self.head = nextNode
        
        
    def top(self):
        return self.head.val
        
        
    def getMin(self):
        return self.head.min
```
- or having two stacks (lists) for storing the numbers and the other storing the min value so far:
```py
class Stack:
    def __init__(self):
        self._stack = []
        
    def push(self, val):
        self._stack.append(val)
        
        
    def pop(self):
        if self.isEmpty():
            return None
        
        return self._stack.pop()
        
        
    def top(self):
        if self.isEmpty():
            return None
        
        return self._stack[-1]
        
    def isEmpty(self):
        return len(self._stack) == 0
    
    def size(self):
        return len(self._stack)

class MinStack:

    def __init__(self):
        self._mainStack = Stack()
        self._minStack = Stack()
        

    def push(self, val: int) -> None:
        self._mainStack.push(val)
        
        if self._minStack.isEmpty() or self._minStack.top() > val:
            self._minStack.push(val)
        else:
            self._minStack.push(self._minStack.top())
        

    def pop(self) -> None:
        self._mainStack.pop()
        self._minStack.pop()
        

    def top(self) -> int:
        return self._mainStack.top()
        

    def getMin(self) -> int:
        if self._minStack.isEmpty():
            return None
        
        return self._minStack.top()
```

## Valid Parantheses
- https://leetcode.com/problems/valid-parentheses/
```py
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        hashMap = {")" : "(", "]" : "[", "}":"{"}
    
        for char in s:
            if char in hashMap:
                if len(stack) == 0:
                    return False

                ele = stack.pop()
                if ele != hashMap[char]:
                    return False

            else:
                stack.append(char)


        return len(stack) == 0
```

## Daily Temperatures
- https://leetcode.com/problems/daily-temperatures/
```py
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        stack = []
        output = [0] * len(temperatures)

        for currentDay, currentTemp in enumerate(temperatures):

            while stack and stack[-1][1] < currentTemp:
                prevDay, _ = stack.pop()
                output[prevDay] = currentDay - prevDay


            stack.append((currentDay, currentTemp))

        return output
```

## Evaluate Reverse Polish Notation
- https://leetcode.com/problems/evaluate-reverse-polish-notation/
```py
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        if not tokens:
            return
        
        stack = []
        def add(a, b):
            return a + b
        
        def minus(a, b):
            return a - b
        
        def multiply(a, b):
            return a * b
        
        def division(a, b):
            return int(a / b)
        
        operatorsMap = {"+": add, "-": minus, "*": multiply, "/": division}
        
        for ele in tokens:
            
            if ele in operatorsMap:
                b, a = stack.pop(), stack.pop()
                stack.append(operatorsMap[ele](int(a), int(b)))
                
            else:
                stack.append(ele)
        
        return stack.pop()
```

# Depth First Search (DFS)

- We don't know if the found path is the shortest path between two vertices.
- Instead of queue in BFS, we use stack (LIFO) in DFS.
- The average time complexity for DFS on a graph is O(V + E), where V is the number of vertices and E is the number of edges. In case of DFS on a tree, the time complexity is O(V), where V is the number of nodes.
- We say average time complexity because a set’s `in` operation has an average time complexity of O(1). If we used a list, the complexity would be higher.
- What if you want to find the shortest path?
Hint: Add one more parameter to indicate the shortest path you have already found.

```py
visited = set() # Set to keep track of visited nodes.
def dfs(visited, graph, node):
    if node not in visited:
        print (node)
        visited.add(node)
        for neighbour in graph[node]:
            dfs(visited, graph, neighbour)
```

```py
# DFS algorithm
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)

    print(start)

    for next in graph[start] - visited:
        dfs(graph, next, visited)
    return visited


graph = {'0': set(['1', '2']),
         '1': set(['0', '3', '4']),
         '2': set(['0']),
         '3': set(['1']),
         '4': set(['2', '3'])}

dfs(graph, '0')
```

## Deep copy of a graph
- https://github.com/SanazME/Algorithm-DataStructure/blob/master/Graph.md#problem-2
```py
"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""
import collections

class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        
        if node is None:
            return node

        visited = {}
        return self.dfs(node, visited)


    def dfs(self, node, visited):
        if node in visited:
            return visited[node]


        copyNode = Node(node.val, [])
        visited[node] = copyNode
        for neighbor in node.neighbors:
            copyNode.neighbors.append(self.dfs(neighbor, visited))
            
        return copyNode
```

## Target Sum
- https://github.com/SanazME/Algorithm-DataStructure/blob/master/Graph.md#problem-2-1


### Two sum
- https://leetcode.com/problems/two-sum


## Binary Tree in-order traversal
- https://leetcode.com/problems/binary-tree-inorder-traversal/
1. DFS recursive:
```py
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        return self.helper(root, result=[])
    
    def helper(self, root: TreeNode, result: List[int]) -> List[int]:
        if root is None:
            return result
        
        self.helper(root.left, result)
        result.append(root.val)
        self.helper(root.right, result)
        
        return result
```

2. DFS stack:
```py
def inorderTraversal(self, root: TreeNode) -> List[int]:
    result=[]
    stack = [(root, False)]

    while stack:
        curNode, visited = stack.pop()

        if curNode:
            if visited:
                result.append(curNode.val)
            else:
                stack.append((curNode.right, False))
                stack.append((curNode, True))
                stack.append((curNode.left, False))

    return result
```

## Decode String
- https://leetcode.com/problems/decode-string/
- We can use two stacks to store the freq and the decoded strings. Instead of pushing the decoded string to the stack character by character, we can append all the characters into the string first and then push the entire string into the stack.

Case 1) if the current character is a digit, append it to the number `k`
Case 2) if the current character is a letter, append it to the `currString`
Case 3) if the current character is a `[`, push `k` and `currString` into stacks and reset those variables
Case 4) if the current character is a `]`: We must begin decoding:
    - We must decode the currentString. Pop `currentK` from the `countStack` and decode the pattern `currentK[currentString]`
    - As the stringStack contains the previously decoded string, pop the `decodedString` from the `stringStack`. Update the `decodedString = decodedString + currentK[currentString]`
```py
class Solution:
    def decodeString(self, s: str) -> str:
        numStack = []
        strStack = []
        
        freq = 0
        currString = ''
        
        for char in s:
            if char.isnumeric():
                freq = freq * 10 + (ord(char) - ord('0'))
                
            elif char.isalpha():
                currString = currString + char
                
            elif char == '[':
                numStack.append(freq)
                strStack.append(currString)
                
                freq = 0
                currString = ''
                
            else:
                decodedString = strStack.pop()
                
                tmp = ''
                for _ in range(numStack.pop()):
                    tmp += currString
                
                decodedString += tmp
                currString = decodedString
                
        return currString
```
- **Time complexity** `O(maxK * n)`, `maxK`: max value of `k` and `n`:  size of string
- **Space complexity** `O(l + m)`, `l`: number of letters, `m`: number of digits

## Flood fill
- https://leetcode.com/problems/flood-fill/

```py
class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
        
        
        
        def dfs(i, j, origColor, visited):
            if i < 0 or i >= len(image) or j < 0 or j >= len(image[0]) or image[i][j] not in [origColor] or (i,j) in visited:
                return
            
            
            visited.add((i,j))
            image[i][j] = color
            dfs(i + 1, j, origColor, visited)
            dfs(i - 1, j, origColor, visited)
            dfs(i, j + 1, origColor, visited)
            dfs(i, j - 1, origColor, visited)
        
        
        dfs(sr, sc, image[sr][sc], set())
        
        return image```


# DFS with stack:
- Given the root of a binary tree and an integer targetSum, return true if the tree has a root-to-leaf path such that adding up all the values along the path equals targetSum.

**DFS with Stack**
```py
def hasPathSum(root, targetSum):
    if not root:
        return False

    sumSofar = root.val
    stack = [(root, 0)]

    while stack:
        print([(node.val, sumAll) for node, sumAll in stack])
        node, localSum = stack.pop()

        currentSum = node.val + localSum

        if not(node.left or node.right):
            if currentSum == targetSum:
                return True

        else:
            if node.left: stack.append((node.left, currentSum))
            if node.right: stack.append((node.right, currentSum))
                
    return False
```
```py
# 1. DFS with stack 2
def hasPathSum(root, targetSum):
    if not root:
        return False
        
    stack = [(root, root.val)]
    
    while stack:
        node, val = stack.pop()
        
        if val == targetSum and not(node.left or node.right):
            return True
            
        if node.left: stack.append((node.left, val + node.left.val))
        if node.right: stack.append((node.right, val + node.right.val))
    return False
```

**Recursive**
```py
def hasPathSum(root, targetSum):
    if not root:
        return False
        
    if root.val == targetSum and not(root.left, root.right):
        return True
        
    return hasPathSum(root.left, targetSum - root.val) or hasPathSum(root.right, targetSum - root.val)
```
**BFS with queue**
```py
def hasPathSum(root, targetSum):
    if not root:
        return False
        
    queue = [(root, targetSum - root.val)]
    
    while queue:
        curr, val = queue.popleft()
        if val == 0 and not(curr.left or curr.right):
            return True
        if curr.left: queue.append((curr.left, val - curr.left.val))
        if curr.right: queue.append((curr.right, val - curr.right.val))
    return False

```

### Path Sum II: https://leetcode.com/problems/path-sum-ii/
**DFS with stack**
```py
def pathSum(self, root, targetSum):
        """
        :type root: TreeNode
        :type targetSum: int
        :rtype: List[List[int]]
        """
        result = []
        if not root:
            return result

        stack = [(root, 0, [])]


        while stack:
            node, localSum, branchList = stack.pop()
            localSum += node.val

            if localSum == targetSum and not(node.left or node.right):
                result.append(branchList + [node.val])

            else:
                if node.right:
                    stack.append((node.right, localSum, branchList + [node.val]))
                if node.left:
                    stack.append((node.left, localSum, branchList + [node.val]))

        return result
```
```py
def pathSum(self, root, targetSum):
        """
        :type root: TreeNode
        :type targetSum: int
        :rtype: List[List[int]]
        """
        result = []
        if not root:
            return result
        
        stack = [(root, targetSum, [root.val])]      
        
        while stack:
            node, sumNodes, branchResult = stack.pop()
                  
            # reaching a leaf node
            if not(node.left or node.right) and node.val == sumNodes:
                result.append(branchResult)    
                
            if node.left: stack.append((node.left, sumNodes - node.val, branchResult+ [node.left.val]))
            if node.right: stack.append((node.right, sumNodes - node.val, branchResult+[node.right.val]))

        return result

```
**Recursive DFS**
```py
def pathSum(self, root, targetSum):
        """
        :type root: TreeNode
        :type targetSum: int
        :rtype: List[List[int]]
        """
        return helper(root, targetSum, [], [])
    
def helper(node, targetSum, branchList, result):
    if node == None:
        return []
    
    if node.val == targetSum and not(node.left or node.right):
        newBranch = branchList + [node.val]
        result.append(newBranch)
        
    if node.left: helper(node.left, targetSum - node.val, branchList + [node.val], result) 
    if node.right: helper(node.right, targetSum - node.val, branchList + [node.val], result)
    
    return result
```
```py
def pathSum(self, root, targetSum):
        """
        :type root: TreeNode
        :type targetSum: int
        :rtype: List[List[int]]
        """
        # Recursive DFS
        if not root:
            return []
        result = []
        self.dfs(root, targetSum, [root.val], result)
        return result
    
    def dfs(self, node, sumNodes, branchResult, result):
        
        if not(node.left or node.right) and sumNodes == node.val:
            result.append(branchResult)
            
        if node.left: self.dfs(node.left, sumNodes - node.val, branchResult + [node.left.val], result)     
        if node.right: self.dfs(node.right, sumNodes - node.val,  branchResult + [node.right.val], result)

```


## Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it is able to trap after raining.
# Example:
# Input: [0,1,0,2,1,0,1,3,2,1,2,1]
# Output: 6

- The brute force solution: for each location, find the max on the left and max on the right of that location and take the min of them
- Optimal solution: calculate the column of water in each location based on the followings:
    - If there was an infinite tall wall on the right end of the array, the water in each location would be the height of max so far on the left of the location - height of the location
    - Now for the locations on the right of the infinite wall, the water in each location coming from right to left is heigth of max so far on the right of the location - height of location.

```py
def trap(self, height):
    """
    :type height: List[int]
    :rtype: int
    """
    size = len(height)
    area = 0

    if size == 0:
        return area

    globalIdx, globalMax = self.findGlobalMax(height)

    # water trap values coming from left to right: find the local max on the left side and calculate the trapped water = (local_max - current_height)*width(=1)
    max_height_local = height[0]
    for i in range(0, globalIdx):
        if height[i] > max_height_local:
            max_height_local = height[i]

        area += (max_height_local - height[i]) * 1

    # water trap values coming from right to left: find the local max on the right side and calculate trapped water = (local_max - current_height)*width
    max_height_local = height[-1]
    for i in range(size - 1, globalIdx, -1):
        if height[i] > max_height_local:
            max_height_local = height[i]
        area += (max_height_local - height[i]) * 1

    return area

def findGlobalMax(self, height):
    maxIdx, maxHeight = 0, height[0]

    for i, val in enumerate(height):
        if val > maxHeight:
            maxIdx = i
            maxHeight = val
    return (maxIdx, maxHeight)
 ```

## Longest consecutive sequence:
- https://leetcode.com/problems/longest-consecutive-sequence/
- **solutions:**
- **1. Brute force**:
- it just considers each number in nums, attempting to count as high as possible from that number using only numbers in nums. After it counts too high (i.e. currentNum refers to a number that nums does not contain), it records the length of the sequence if it is larger than the current best. The algorithm is necessarily optimal because it explores every possibility.
```py
def longestConsecutive(nums):
    if not nums:
        return 0

    longest = 0
    for num in nums:
        curr = num
        local_streak = 1
        
        while curr + 1 in nums:
            local_streak += 1
            curr += 1
        
        longest = max(longest, local_streak)
    return longest
```
- **Time complexity**: O(n^3) - the for loop: n, the while loop is n and the `in` operator in `while` is O(n)..
- **Space complexity** O(1).

**2. Define nums as a set so we do lookup in in O(1) time:**
- First turn the input into a set of numbers. That takes O(n) and then we can ask in O(1) whether we have a certain number. Then go through the numbers. If the number x is the start of a streak (i.e., x-1 is not in the set), then test y = x+1, x+2, x+3, ... and stop at the first number y not in the set. The length of the streak is then simply y-x and we update our global best with that. Since we check each streak only once, this is overall **O(n)**.
```py
def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums = set(nums)
        result = 0
        
        for x in nums:
            if x - 1 not in nums:
                y = x + 1
                while y in nums:
                    y += 1
                result = max(result, y - x)
        return result
```
## 
data structure (min heap, max heap and Priority Queue): 
- Look at readme in tree: https://github.com/SanazME/Algorithm-DataStructure/blob/master/trees/README.md#tree
- https://www.youtube.com/watch?v=HqPJF2L5h9U
- https://www.programiz.com/dsa/priority-queue
- Python heap: `heappush, heappop...` for min heap

## Spiral Array
- https://leetcode.com/problems/spiral-matrix/

```py
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        rows = len(matrix)
        cols = len(matrix[0])
        
        output = []
        
        minRow = minCol = 0
        maxRow = rows - 1
        maxCol = cols - 1
        
        while len(output) < rows * cols:
            
            if minRow <= maxRow and minCol <= maxCol:
                i = minRow
                for j in range(minCol, maxCol + 1):
                    output.append(matrix[i][j])
                minRow += 1

                # if minRow > maxRow:
                #     break

            if minRow <= maxRow and minCol <= maxCol:
                j = maxCol
                for i in range(minRow, maxRow + 1):
                    output.append(matrix[i][j])
                maxCol -= 1

                # if minCol > maxCol:
                #     break

            if minRow <= maxRow and minCol <= maxCol:
                i = maxRow
                for j in range(maxCol, minCol - 1, -1):
                    output.append(matrix[i][j])
                maxRow -= 1

                # if minRow > maxRow:
                #     break

            if minRow <= maxRow and minCol <= maxCol:
                j = minCol
                for i in range(maxRow, minRow - 1, -1):
                    output.append(matrix[i][j])
                minCol += 1

                # if minCol > maxCol:
                #     break
            
        
        return output
```
## Pascal Triangle
- https://leetcode.com/problems/pascals-triangle/

- our output list will store each row as a sublist
- the first and last element of each sublist is 1.
- we then can calculate each element in between based on pervious sublist elements
```py
def generate(numRows):
    triangle = [[1]]
    
    if numRows > 1:
        for row in range(1, numRows):
            sublist = [0] * (row + 1)
            sublist[0] = sublist[-1] = 1
            
            for k in range(1, row):
                sublist[k] = triangle[row - 1][k] + triangle[row - 1][k - 1]
            
            triangle.append(sublist)

    
    return triangle
```

## Minimum Size Subarray Sum
- https://leetcode.com/problems/minimum-size-subarray-sum/
**Algorithm**
1. Initialize `left` pointer to 0
2. Iterate over the array:
    - Add to the sum
    - while sum is larger than the target:
        - update the answer
        - remove from the sum index....
```py
def minSubarray(nums, target):
    maxVal = max(nums)
    if maxVal >= target:
        return 1
    
    if len(nums) == 0:
        return 0
    
    left = 0
    sumSoFar = 0
    countSoFar = 0
    minCount = float('Inf')
    
    for i in range(len(nums)):
        sumSoFar += nums[i]
        while sumSoFar >= target:
            minCount = min(minCount, i - left + 1)
            sumSoFar -= nums[left]
            left += 1
            
    if minCount != float('Inf'):
        return minCount
    else:
        return 0
```
## Rotate Array
- https://leetcode.com/problems/rotate-array

**Solution 1**:
with extra space
```py
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        k = k % len(nums)
        if k == 0 or len(nums) == 0:
            return nums
        
        output = [0] * len(nums)
        
        for i in range(len(nums)):
            output[(i + k) % len(nums)] = nums[i]
            
        nums[:] = output
```

without extra space and chaning in-place:
We can directly place every number of the array at its required correct position. But if we do that, we will destroy the original element. Thus, we need to store the number being replaced in a `temp` variable. Then, we can place the replaced number `temp` at its correct position and so on, n times, where n is the length of array. We have chosen nn to be the number of replacements since we have to shift all the elements of the array(which is n).
But, there could be a problem with this method, if `n % k = 0` where `k = k % n` (since a value of k larger than n eventually leads to a k equivalent to `k % n`). In this case, while picking up numbers to be placed at the correct position, we will eventually reach the number from which we originally started. Thus, in such a case, when we hit the original number's index again, we start the same process with the number following it.

Now let's look at the proof of how the above method works. Suppose, we have n as the number of elements in the array and k is the number of shifts required. Further, assume `n %k = 0`. Now, when we start placing the elements at their correct position, in the first cycle all the numbers with their index i satisfying `i % k = 0` get placed at their required position. This happens because when we jump k steps every time, we will only hit the numbers k steps apart. We start with index `i = 0`, having `i % k = 0`. Thus, we hit all the numbers satisfying the above condition in the first cycle. When we reach back the original index, we have placed `n/k` elements at their correct position, since we hit only that many elements in the first cycle. Now, we increment the index for replacing the numbers. This time, we place other `n/k` elements at their correct position, different from the ones placed correctly in the first cycle, because this time we hit all the numbers satisfy the condition `i % k = 1`. 
When we hit the starting number again, we increment the index and repeat the same process from `i = 1` for all the indices satisfying `i % k == 1`. This happens till we reach the number with the index `i % k = 0` again, which occurs for `i=k`. We will reach such a number after a total of k cycles. Now, the total count of numbers exclusive numbers placed at their correct position will be `k * n/k = n`. Thus, all the numbers will be placed at their correct position.
```py
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        n = len(nums)
        k %= n
        
        start = count = 0
        while count < n:
            current, prev = start, nums[start]
            while True:
                next_idx = (current + k) % n
                nums[next_idx], prev = prev, nums[next_idx]
                current = next_idx
                count += 1
                
                if start == current:
                    break
            start += 1
```

## Design a circular queue
- A more efficient way is to use a circular queue. Specifically, we may use a fixed-size array and two pointers to indicate the starting position and the ending position. And the goal is to reuse the wasted storage we mentioned previously.
- https://leetcode.com/explore/learn/card/queue-stack/228/first-in-first-out-data-structure/1396/
- Create a circular queue with enQueue, deQueue, Front and Rear methods and any other methods that is needed.

```py
class MyCircularQueue:
    def __init__(self, size):
        self.size = size
        self.head = -1
        self.tail = -1
        self.count = 0
        self.arr = [None] * self.size
        
        
    def enQueue(self, val):
        if self.isFull():
            return False
        
        # move tail index one unit forward
        if self.head == -1:
            self.head, self.tail = 0, 0
        
        else:
            self.tail = (self.tail + 1) % self.size
        
        self.arr[self.tail] = val     
        self.count += 1
                
        return True
            
        
    def deQueue(self):
        if self.isEmpty():
            return False
        
        
        self.head = (self.head + 1) % self.size
        self.count -= 1

        return True
        
        
    
    def isFull(self):
        if self.count == self.size:
            return True
        return False
        
    def isEmpty(self):
        return self.count == 0
    
    def Front(self):
        if self.isEmpty():
            return -1
        return self.arr[self.head]
        
    def Rear(self):
        if self.isEmpty():
            return -1
        return self.arr[self.tail]
```

OR with only one `head` pointer and deducing `tail` index from `head` and `count`:
```py
class MyCircularQueue:
    def __init__(self, k):
        self.capacity = k
        self.head = 0
        self.count = 0
        self.queue = [None] * self.capacity
    
    def enQueue(self, val):
        if self.isFull():
            return False
        
        # move tail index one unit forward
        self.queue[(self.head + self.count) % self.capacity] = val
        self.count += 1
                
        return True
            
    def deQueue(self):
        if self.isEmpty():
            return False
        
        self.head = (self.head + 1) % self.capacity
        self.count -= 1

        return True
    
    def isFull(self):
        return self.count == self.capacity
            
    def isEmpty(self):
        return self.count == 0
    
    def Front(self):
        if self.isEmpty():
            return -1
        return self.queue[self.head]
        
    def Rear(self):
        if self.isEmpty():
            return -1
        return self.queue[(self.head + self.count - 1) % self.capacity]
```
- This solution is not thread safe and there can be a race condition for incrementing the counter among different threads. To implement a safe thread solution, we need to use lock(). For example for enQueue method:
```py
from threading import Lock

class MyCircularQueue:
    def __init__(self, k):
        self.capacity = k
        self.head = 0
        self.count = 0
        self.queue = [None] * self.capacity
        self.queueLock = Lock()
    
        
        
    def enQueue(self, val):
        
        with self.queueLock:
            if self.isFull():
                return False

            # move tail index one unit forward
            self.queue[(self.head + self.count) % self.capacity] = val
            self.count += 1
                
        return True
            
        
    def deQueue(self):
        
        with self.queueLock:
            if self.isEmpty():
                return False


            self.head = (self.head + 1) % self.capacity
            self.count -= 1

        return True
        
        
    
    def isFull(self):
        return self.count == self.capacity
            
    def isEmpty(self):
        return self.count == 0
    
    def Front(self):
        if self.isEmpty():
            return -1
        return self.queue[self.head]
        
    def Rear(self):
        if self.isEmpty():
            return -1
        return self.queue[(self.head + self.count - 1) % self.capacity]
```

## Moving Average from Data stream
- https://leetcode.com/problems/moving-average-from-data-stream/
- We can use an array with ever growing in size as the stream of data is coming:

```py
class MovingAverage:

    def __init__(self, size: int):
        self.size = size
        self.count = 0
        self.arr = []
        self.start = 0
        self.sumWindow = 0
        

    def next(self, val: int) -> float:
        if self.count < self.size:
            self.sumWindow += val
            self.arr.append(val)
            self.count += 1
        else:
            self.arr.append(val)
            self.sumWindow -= self.arr[self.start]
            self.sumWindow += val
            self.start += 1

        return (self.sumWindow * 1.0) / self.count
```
- The better approach is to use **circular queue** so that the space complexity is `O(N)` ( size of window) and time complexity is `O(1)`:
```py
class MovingAverage:

    def __init__(self, size: int):
        self.size = size
        self.count = 0
        self.queue = [0] * self.size
        self.head = 0
        self.sumWindow = 0
        

    def next(self, val: int) -> float:
        
        idx = (self.head + self.count) % self.size
        
        if self.count < self.size:
            self.count += 1
        else:
            self.sumWindow -= self.queue[idx]
            self.head = (self.head + 1) % self.size
            
        self.queue[idx] = val
        self.sumWindow += val

        return (self.sumWindow * 1.0) / self.count
```

## 1. Subtree of Another Tree
- https://leetcode.com/problems/subtree-of-another-tree/description/

### Solution 1: DFS
- Let's consider the most naive approach first. We can traverse the tree rooted at root (using Depth First Search) and for each node in the tree, check if the "tree rooted at that node" is identical to the "tree rooted at subRoot". If we find such a node, we can return true. If traversing the entire tree rooted at root doesn't yield any such node, we can return false.

Since we have to check for identicality, again and again, we can write a function isIdentical which takes two roots of two trees and returns true if the trees are identical and false otherwise.

Checking the identicality of two trees is a classical task. We can use the same approach as the one in Same Tree Problem. We can traverse both trees simultaneously and

   - if any of the two nodes being checked is null, then for trees to be identical, both the nodes should be null. Otherwise, the trees are not identical.
   - if both nodes are non-empty. Then for the tree to be identical, ensure that:
        - values of the nodes are the same
        - left subtrees are identical
        - right subtrees are identical

```py
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def isSubtree(self, root, subRoot):
        """
        :type root: TreeNode
        :type subRoot: TreeNode
        :rtype: bool
        """
        
        if self.isIdentical(root, subRoot): return True

        if root == None:
            return False

        return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)

        
    def isIdentical(self, subtree1, subtree2):
        if subtree1 == None or subtree2 == None:
            return subtree1 == subtree2 == None

        return (subtree1.val == subtree2.val and 
        self.isIdentical(subtree1.left, subtree2.left) and
        self.isIdentical(subtree1.right, subtree2.right))
```

**Time complexity**:
- `O(M * N)`. For every `N` node in the tree, we check if the tree rooted at node is identical to subRoot. This check takes `O(M)` time, where `M` is the number of nodes in subRoot. Hence, the overall time complexity is `O(M * N)`.

**Space complexity**
- There will be at most `N` recursive call to dfs ( or isSubtree). Now, each of these calls will have `M` recursive calls to isIdentical. Before calling isIdentical, our call stack has at most `O(N)` elements and might increase to `O(N + M)` during the call. After calling isIdentical, it will be back to at most `O(N)` since all elements made by isIdentical are popped out. Hence, the maximum number of elements in the call stack will be `M+N`.

### Solution 2: Hash Table
- It turns out that tree comparison is expensive. In the very first approach, we need to perform the comparison for at most `N` nodes, and each comparison cost `O(M)`. If we can somehow reduce the cost of comparison, then we can reduce the overall time complexity

You may recall that the cost of comparison of two integers is constant. As a result, if we can somehow transform the subtree rooted at each node to a unique integer, then we can compare two trees in constant time.

_Is there any way to transform a tree into an integer?
Yes, there is. We can use the concept of Hashing.
_
We want to hash (map) each subtree to a unique value. We want to do this in such a way that if two trees are identical, then their hash values are equal. And, if two trees are not identical, then their hash values are not equal. This hashing can be used to compare two trees in `O(1)` time.

We will build the hash of each node depending on the hash of its left and right child. The hash of the root node will represent the hash of the whole tree because to build the hash of the root node, we used (directly, or indirectly) the hash values of all the nodes in its subtree.

If any node in "tree rooted at root" has hash value equal to the hash value of "tree rooted at subRoot", then "tree rooted at subRoot" is a subtree of "tree rooted at root", provided our hashing mechanism maps nodes to unique values.

There can be multiple ways of hashing the tree. We want to use that mechanism which is

 - Simple to calculate
- Efficient
- Has minimum spurious hits
Spurious Hits: If hash values of two trees are equal, and still they are not identical, then we call it a spurious hit. A spurious hit is a case of False Positive.

One can use any hashing function which guarantees minimum spurious hits and is calculated in `O(1)` time. We will use the following hashing function.

- If it's a null node, then hash it to 3. (We can use any prime number here)
- Else,
    - left shift the hash value of the left node by some fixed value.
    - left shift hash value of right node by 1
    - add these shifted values with this `node->val` to get the hash of this node.

**Please note that one should avoid concatenating strings for hash value purposes because it will take `O(N)` time to concatenate strings.**

To ensure minimum spurious hits, we can map each node to two hash values, thus getting one hash pair for each node. Trees rooted at s and Tree rooted at t will have the same hash pair iff they are identical, provided our hashing technique maps nodes to unique hash pairs.

One can read more about this in Editorial of 28. Find the Index of the First Occurrence of a String in Another String.

```py
class Solution:
    def isSubtree(self, root: TreeNode, subRoot: TreeNode) -> bool:

        MOD_1 = 1_000_000_007
        MOD_2 = 2_147_483_647

        def hash_subtree_at_node(node, need_to_add):
            if node is None:
                return (3, 7)

            left = hash_subtree_at_node(node.left, need_to_add)
            right = hash_subtree_at_node(node.right, need_to_add)

            left_1 = (left[0] << 5) % MOD_1
            right_1 = (right[0] << 1) % MOD_1
            left_2 = (left[1] << 7) % MOD_2
            right_2 = (right[1] << 1) % MOD_2

            hashpair = ((left_1 + right_1 + node.val) % MOD_1,
                        (left_2 + right_2 + node.val) % MOD_2)

            if need_to_add:
                memo.add(hashpair)

            return hashpair

        # List to store hashed value of each node.
        memo = set()

        # Calling and adding hash to List
        hash_subtree_at_node(root, True)

        # Storing hashed value of subRoot for comparison
        s = hash_subtree_at_node(subRoot, False)

        # Check if hash of subRoot is present in memo
        return s in memo
```

**Time Complexity** `O(M+N)`:
We are traversing the tree rooted at root in `O(N)` time. We are also traversing the tree rooted at subRoot in `O(M)` time. For each node, we are doing constant time operations. After traversing, for lookup we are either doing `O(1)` operations, or `O(N)` operations. Hence, the overall time complexity is `O(N+M)`

**Space Complexity** `O(M+N)`:
We are using memo to store the hash pair of each node in the tree rooted at root. Hence, for this, we need `O(N)` space.
Moreover, since we are using recursion, the space required for the recursion stack will be `O(N)` for `hashSubtreeAtNode(root, true)` and `O(M)` for `hashSubtreeAtNode(subRoot, false)`.
Hence, overall space complexity is` O(M+N)`.

## 2. Kth Largest Element in a Stream
- https://leetcode.com/problems/kth-largest-element-in-a-stream/description/
```py
from heapq import *

class KthLargest:
    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.heap = nums
        heapq.heapify(self.heap)
        
        while len(self.heap) > k:
            heapq.heappop(self.heap)

    def add(self, val: int) -> int:
        heapq.heappush(self.heap, val)
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)
        return self.heap[0]
```

**Time Complexity**
Given `N` as the length of nums and `M` as the number of calls to add(),
- https://www.youtube.com/watch?v=HqPJF2L5h9U

**Time complexity**: 1O(N⋅log⁡(N)+M⋅log⁡(k))`

The time complexity is split into two parts. 
- First, the constructor needs to turn nums into a heap of size `k`. **In Python, `heapq.heapify()` can turn nums into a heap in `O(N)` time**. Then, we need to remove from the heap until there are only `k` elements in it, which means removing `N - k` elements. Since k can be, say 1, in terms of big O this is `N` operations, with each operation costing `log⁡(N)`. Therefore, the constructor costs `O(N+N⋅log⁡(N))=O(N⋅log⁡(N))`.
    - Push into a heap (sift down):  `O(log N)`, sift down: height of the heap (complete binary tree) which is `log N`
    - Pop (and sift up): `O(log N)`, sift up: height of the heap (complete binary tree) which is `log N`

- Next, every call to add() involves adding an element to heap and potentially removing an element from heap. Since our heap is of size k, every call to add() at worst costs O(2∗log⁡(k))=O(log⁡(k))O(2 * \log(k)) = O(\log(k))O(2∗log(k))=O(log(k)). That means M calls to add() costs `O(M⋅log⁡(k))`.

**Space complexity:** `O(N)`


## 3. Design a HashSet
- https://leetcode.com/problems/design-hashset/

**hash function:** the goal of the hash function is to assign an address to store a given value. Ideally, each unique value should have a unique hash value.

**collision handling:** since the nature of a hash function is to map a value from a space A into a corresponding value in a smaller space B, it could happen that multiple values from space A might be mapped to the same value in space B. This is what we call collision. Therefore, it is indispensable for us to have a strategy to handle the collision.

Overall, there are several strategies to resolve the collisions:
- **Separate Chaining:** for values with the same hash key, we keep them in a bucket, and each bucket is independent of each other.

- **Open Addressing:** whenever there is a collision, we keep on probing on the main space with certain strategy until a free slot is found.

- **2-Choice Hashing:** we use two hash functions rather than one, and we pick the generated address with fewer collision

we focus on the strategy of **separate chaining**. Here is how it works overall.

- Essentially, the primary storage underneath a HashSet is a continuous memory as Array. Each element in this array corresponds to a bucket that stores the actual values.

- Given a value, first we generate a key for the value via the hash function. The generated key serves as the index to locate the bucket.

- Once the bucket is located, we then perform the desired operations on the bucket, such as add, remove and contains.

### Approach 1: LinkedList as Bucket
The common choice of hash function is the `modulo` operator, i.e. `hash=value mod  base`. Here, the `base` of modulo operation would determine the number of buckets that we would have at the end in the HashSet. it is generally advisable to use a prime number as the base of modulo, e.g. `769`, in order to reduce the potential collisions.

![linked list](705_linked_list.png "Linked List Buckets")

As to the design of bucket, again there are several options. One could simply use another Array as bucket to store all the values. However, one drawback with the Array data structure is that it would take `O(N)` time complexity to remove or insert an element, rather than the desired `O(1)`.

Since for any update operation, we would need to scan the entire bucket first to avoid any duplicate, a better choice for the implementation of bucket would be the LinkedList, which has a **constant time complexity for the insertion as well as deletion, once we locate the position to update.**

```py

class MyHashSet(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.keyRange = 769
        self.bucketArray = [Bucket() for i in range(self.keyRange)]

    def _hash(self, key):
        return key % self.keyRange

    def add(self, key):
        """
        :type key: int
        :rtype: None
        """
        bucketIndex = self._hash(key)
        self.bucketArray[bucketIndex].insert(key)

    def remove(self, key):
        """
        :type key: int
        :rtype: None
        """
        bucketIndex = self._hash(key)
        self.bucketArray[bucketIndex].delete(key)

    def contains(self, key):
        """
        Returns true if this set contains the specified element
        :type key: int
        :rtype: bool
        """
        bucketIndex = self._hash(key)
        return self.bucketArray[bucketIndex].exists(key)


class Node:
    def __init__(self, value, nextNode=None):
        self.value = value
        self.next = nextNode

class Bucket:
    def __init__(self):
        # a pseudo head
        self.head = Node(0)

    def insert(self, newValue):
        # if not existed, add the new element to the head.
        if not self.exists(newValue):
            newNode = Node(newValue, self.head.next)
            # set the new head.
            self.head.next = newNode

    def delete(self, value):
        prev = self.head
        curr = self.head.next
        while curr is not None:
            if curr.value == value:
                # remove the current node
                prev.next = curr.next
                return
            prev = curr
            curr = curr.next

    def exists(self, value):
        curr = self.head.next
        while curr is not None:
            if curr.value == value:
                # value existed already, do nothing
                return True
            curr = curr.next
        return False


# Your MyHashSet object will be instantiated and called as such:
# obj = MyHashSet()
# obj.add(key)
# obj.remove(key)
# param_3 = obj.contains(key)
```

**Time Complexity**
- `O(N/K)` where `N` is the number of all possible values and `K` is the number of predefined buckets which 769
    - assuming that the values are evenly distributed, we could consider the average size of a bucket is `N/K`
    - Since for each operation, in the worst case, we would need to scan the entire bucket, the time complexity is `O(N/K)`

**Space Complexity**
- `O(K + M)` where `K` is the number of predefined buckets and `M` is the number of unique values that have been inserted into the HashSet.

### Approach 2: Binary Search Tree
In the above approach, one of the drawbacks is that we have to scan the entire linkedlist in order to verify if a value already exists in the bucket (i.e. the lookup operation).

To optimize the above process, one of the strategies could be that we maintain a **sorted list as the bucket**. With the sorted list, we could obtain the `O(log⁡N)` time complexity for the lookup operation, with the binary search algorithm, rather than a linear `O(N)` complexity as in the above approach.

On the other hand, if we implement the sorted list in a continuous space such as Array, it would incur a linear time complexity for the update operations (e.g. insert and delete), since we would need to shift the elements.

_So the question is can we have a data structure that have `O(log⁡N)` time complexity, for the operations of search, insert and delete ?_
The answer is yes, with **Binary Search Tree (BST)**. Thanks to the properties of BST, we could optimize the time complexity of our first approach with LinkedList.
![BST](705_BST.png "BST")

One could build upon the implementation of first approach for our second approach, by applying the Façade design pattern.

_We have already defined a façade class (i.e. bucket) with three interfaces (exists, insert and delete), which hides all the underlying details from its users (i.e. HashSet)._

So we can keep the bulk of the code, and simply modify the implementation of bucket class with BST. 

First solve the following for BST:
- [Search in BST](https://leetcode.com/articles/search-in-a-bst/)
- [Insert in BST](https://leetcode.com/articles/insert-into-a-bst/)
- [Delete in BST](https://leetcode.com/articles/delete-node-in-a-bst)

**3 important facts to know about BST:**
1. In-order traversal of a BST is a sorted array in ascending order
```py
def inorder(root: Optional[TreeNode]) -> List:
    return inorder(root.left) + [root.val] + inorder(root.right) if root else []
```

2. Successor = "after node", i.e. the next node, or the smallest node after the current one. To find the successor: **go to the right once
and then as many times to the left as you could.**
```py
def successor(root: TreeNode) -> TreeNode:
    root = root.right
    while root.left:
        root = root.left
    return root
```

3. Predecessor = "before node", i.e. the previous node, or the largest node before the current one. To find the predecessor: **go to the left and then as many times to the right as you could**
```py
def predecessor(root: TreeNode) -> TreeNode:
    root = root.left
    while root.right:
        root = root.right
    return root
```


```py
class MyHashSet:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.keyRange = 769
        self.bucketArray = [Bucket() for i in range(self.keyRange)]

    def _hash(self, key) -> int:
        return key % self.keyRange

    def add(self, key: int) -> None:
        bucketIndex = self._hash(key)
        self.bucketArray[bucketIndex].insert(key)

    def remove(self, key: int) -> None:
        """
        :type key: int
        :rtype: None
        """
        bucketIndex = self._hash(key)
        self.bucketArray[bucketIndex].delete(key)

    def contains(self, key: int) -> bool:
        """
        Returns true if this set contains the specified element
        :type key: int
        :rtype: bool
        """
        bucketIndex = self._hash(key)
        return self.bucketArray[bucketIndex].exists(key)

class Bucket:
    def __init__(self):
        self.tree = BSTree()

    def insert(self, value):
        self.tree.root = self.tree.insertIntoBST(self.tree.root, value)

    def delete(self, value):
        self.tree.root = self.tree.deleteNode(self.tree.root, value)

    def exists(self, value):
        return (self.tree.searchBST(self.tree.root, value) is not None)

class TreeNode:
    def __init__(self, value):
        self.val = value
        self.left = None
        self.right = None

class BSTree:
    def __init__(self):
        self.root = None

    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        if root is None or val == root.val:
            return root

        return self.searchBST(root.left, val) if val < root.val \
            else self.searchBST(root.right, val)

    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
        if not root:
            return TreeNode(val)

        if val > root.val:
            # insert into the right subtree
            root.right = self.insertIntoBST(root.right, val)
        elif val == root.val:
            return root
        else:
            # insert into the left subtree
            root.left = self.insertIntoBST(root.left, val)
        return root

    def successor(self, root):
        """
        One step right and then always left
        """
        root = root.right
        while root.left:
            root = root.left
        return root.val

    def predecessor(self, root):
        """
        One step left and then always right
        """
        root = root.left
        while root.right:
            root = root.right
        return root.val

    def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
        if not root:
            return None

        # delete from the right subtree
        if key > root.val:
            root.right = self.deleteNode(root.right, key)
        # delete from the left subtree
        elif key < root.val:
            root.left = self.deleteNode(root.left, key)
        # delete the current node
        else:
            # the node is a leaf
            if not (root.left or root.right):
                root = None
            # the node is not a leaf and has a right child
            elif root.right:
                root.val = self.successor(root)
                root.right = self.deleteNode(root.right, root.val)
            # the node is not a leaf, has no right child, and has a left child
            else:
                root.val = self.predecessor(root)
                root.left = self.deleteNode(root.left, root.val)

        return root

# Your MyHashSet object will be instantiated and called as such:
# obj = MyHashSet()
# obj.add(key)
# obj.remove(key)
# param_3 = obj.contains(key)
```
**Time complexity**
- `O(log N/K)`, `N` is the number of possible values and `K` is the number of predefined buckets which is **769**
- assuming the values are evenly distributed, we could consider that the average size of bucket is `N/K`
- when we traverse the BST, we are conducting binary search: `O(log N/K)`

**Space complexity**
- `O(N + K)` where `K` is the number of predefined buckets and `M` is the number of unique values that have been inserted into the HashSet.

## Search in BST:
**- recursive:**
```py
def searchBST(root, val):
    if root is None or root.val == val:
        return root
    if root.val > val:
        return searchBST(root.left, val)
    elif root.val < val:
        return searchBST(root.right, val)
```
**- iterative:**
```py
def searchBST(root, val):
    if root is None:
        return
    
    stack = [root]
    
    while stack:
        curr = stack.pop()
        if curr.val == val:
            return curr
        if val > curr.val and curr.right:
            stack.append(curr.right)
        elif val < curr.val and curr.left:
            stack.append(curr.left)
    
    return


def searchBST2(root, val):
    while root and root.val != key:
        if root.val > key:
            root = root.left
        else:
            root = root.right
    return root
```

## Insert in BST:
```py
def addBST(root, val):
    if root is None:
        return Node(val)
    
    if root.val > val:
        root.left = addBST(root.left, val)
    elif root.val < val:
        root.right = addBST(root.right, val)
        
    return root
```
**Iterative:**
```py
class Solution:
    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
        node = root
        while node:
            # insert into the right subtree
            if val > node.val:
                # insert right now
                if not node.right:
                    node.right = TreeNode(val)
                    return root
                else:
                    node = node.right
            # insert into the left subtree
            else:
                # insert right now
                if not node.left:
                    node.left = TreeNode(val)
                    return root
                else:
                    node = node.left
        return TreeNode(val)
```


## Delete in BST:
There are 3 cases to cover:
1. node is a leaf node and we can just delete it.
2. node has a right child:
    - we find it's successor
    - copy the value of successor on the node (replace node with its successor)
    - recursively remove its successor node on the right subtree of the node
3. node has no right child (meaning that its successor is somewhere on the left subtree of its parents that we have not visited), instead we find its predecessor in its left children:
    - find its predecessor
     - copy the value of predecessor on the node (replace node with its predecessor)
    - recursively remove its predecessor node on the left subtree of the node
   
```py
class Solution:
    # One step right and then always left
    def successor(self, root: TreeNode) -> int:
            root = root.right
            while root.left:
                root = root.left
            return root.val
        
    # One step left and then always right
    def predecessor(self, root: TreeNode) -> int:
        root = root.left
        while root.right:
            root = root.right
        return root.val

    def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
        if not root:
            return None

        # delete from the right subtree
        if key > root.val:
            root.right = self.deleteNode(root.right, key)
        # delete from the left subtree
        elif key < root.val:
            root.left = self.deleteNode(root.left, key)
        # delete the current node
        else:
            # the node is a leaf
            if not (root.left or root.right):
                root = None
            # the node is not a leaf and has a right child
            elif root.right:
                root.val = self.successor(root)
                root.right = self.deleteNode(root.right, root.val)
            # the node is not a leaf, has no right child, and has a left child    
            else:
                root.val = self.predecessor(root)
                root.left = self.deleteNode(root.left, root.val)
                        
        return root
```
