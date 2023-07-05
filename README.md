# Algorithm-DataStructure

- **Leetcode top patterns:** https://seanprashad.com/leetcode-patterns/
- **IGotAnOffer system design interviews: https://igotanoffer.com/blogs/tech/system-design-interviews
- **Edabit: https://edabit.com/challenges**

- **14 Patterns to Ace Any Coding Interview Question**: https://hackernoon.com/14-patterns-to-ace-any-coding-interview-question-c5bb3357f6ed

- **Intro to Trie**: https://leetcode.com/discuss/general-discussion/1066206/introduction-to-trie

- **For recommanded problems check my watchlist and forks in leetcode**: https://leetcode.com/discuss/general-discussion/1058072/LeetCode-Advised-Problems-Sorted-by-Topics-and-Directions

- **Dynamic progeamming categoris**: https://leetcode.com/discuss/general-discussion/1050391/Must-do-Dynamic-programming-Problems-Category-wise

- **A noob's guid to Dijkstra's algorithm**: https://leetcode.com/discuss/general-discussion/1059477/A-noob's-guide-to-Djikstra's-Algorithm

- **Youtube Tusher Roy**: https://www.youtube.com/user/tusharroy2525

- **Youtube KA education (I have his udemy course) - interview qs**: https://www.youtube.com/channel/UCvHXUZ7P4wWQ-8LCAg_anQA

- **Youtube Gaurav Sen (System Design)**: https://www.youtube.com/channel/UCRPMAqdtSgd0Ipeef7iFsKw

- **Abdul Bari**: https://www.youtube.com/channel/UCZCFT11CWBi3MHNlGf019nw

- **How to Design a Web Application: Software Architecture 101**: https://hackernoon.com/how-to-design-a-web-application-software-architecture-101-eecy36o5
- **Top 10 System Design Interview Questions for Software Engineers**: https://hackernoon.com/top-10-system-design-interview-questions-for-software-engineers-8561290f0444
- **Top 5 Concurrency Interview Questions for Software Engineers**: https://hackernoon.com/top-5-concurrency-interview-questions-for-software-engineers-x48i30qu

- **Leetcode picks for algorithms & Design patterns**: https://leetcode.com/discuss/general-discussion/1041234/Become-LeetCode's-pick!-Win-LeetCoins-and-LeetCode-goodies?ref=site
    - **System Design: Designing a distributed Job Scheduler**: https://leetcode.com/discuss/general-discussion/1082786/System-Design%3A-Designing-a-distributed-Job-Scheduler-or-Many-interesting-concepts-to-learn

-**Study program - topics (6 months)**: https://leetcode.com/discuss/general-discussion/1129503/Powerful-studying-program-for-Beginners-and-Intermediate-levels.-All-common-mistakes-analyzed

- **Time complexity of search data structures: https://en.wikipedia.org/wiki/Search_data_structure**

**Serialization and Deserialization**
- Serialization is the process of converting a data structure or object into a sequence of bits so that it can be saved in a file or memory buffer, or transmitted across a network connection.

**Tricks to remember**
- To get digits of a number (base 10) from right to left, use modulu (`%`):
```py
n = 196
while n > 0:
    digit-from-right-to-left = n % 10
    n = n // 10

```

**Useful Python methods**
1. `enumerate()` for enumerate :
```python
for count, value in enumerate(values):
    print(count, value)
0 a
1 b
2 c
    
for count, value in enumerate(values, start=1): // the count starts from number 1 
    print(count, value)
1 a
2 b
3 c
```

2. To compare a number with Inf or -Inf map float to string:
```py
x = -float('Inf')
x < 1 # True
```

3. To create default dict to supply missing values:
```py
fron collections import defaultdict
s = [('yellow', 1), ('blue', 2), ('yellow', 3), ('blue', 4), ('red', 1)]
d = defaultdict(list)
for k, v in s:
    d[k].append(v)
    
s = 'mississippi'
d = defaultdict(int)
for k in s:
    d[k] += 1
```

3.1. Another way to create a dictionary instead of using `defaultdict` or just `{}` is to use `dic.get(ele, defaultVal)`:
```py
dic = {}
for ele in list_a:
    dic[ele] = dic.get(ele, 0) + 1
```

4. To round toward zero in integers (positive or negative):
```py
return a//b if a*b>0 else (a+(-a%b))//b
```

5. There is no switch statement, instead we can use dictionary:
```py
def plus(x,y):
        return x + y
def minus(x,y):
    return x - y
def product(x,y):
    return x*y
def division(x,y):
    return round(x//y)
    
operations = {'+': plus,
              '-': minus,
              '/': division,
              '*': product}
              
operations['+'](23,12)
```

6. To add, remove item in a set and to clear a set:
```py
hashset = set()
hashset.add(3)
hashset.add(5)
hashset.update(result) # to add update multiple elements, result can be a list, tuples or strings
hashset.remove(3)
hashset.clear() # ()
```

6.1 For hash map:
```py
hashMap = {1: 4, 5: 6}

del hasMap[1] # delete an element 

if 2 not in hashMap: # check the existence of a key

for key in hashMap:
for key in hashMap.keys():
for val in hashMap.values():
for key, val in hashMap.items():

hashMap.keys() # get all keys

hashMap.clear() # clear the hashMap {}

```

7. To find a number in an array that does not repeated twice (all other numbers do) and to not use extra space, we can use XOR (exclusive OR) operation:
```py
a XOR a = 0
a XOR 0 = a
XOR is associative
so if we XOR all the numbers the non repeated number will be returned: 1 XOR 4 XOR 1 XOR 2 XOR 2 = (1 XOR 1 ) XOR 4 XOR (2 XOR 2) = 0 XOR 4 = 4
```
8. To reverse loop. It loops from the last element up to idx1 index:
```py
for i in range(len(arr) - 1, idx1, -1)
```
9. To get a digits of a number (17988):
```py
digits = []
while n > 0:
    digits.append(n%10)
    n = n//10
print(digits)
```
10. For list comprehension: 
```py
result = [num**2 for num in numList]
```

11. We can have `else` branch for `while`:
```py
while ():

else:
```

12. we can add up boolean variables as integer:
```py
x = (5==5) + (4==4) + (3==2) = 1 + 1 + 0 = 2
```

13. To define a global variable in a class, we define an init function and define a global variable:
```py
class Solution(object):
    def __init__(self):
        self.memo = dict()
```

14. Time complexity of list, array and dictionary for different operations and functions: https://wiki.python.org/moin/TimeComplexity
15. Catalan number: 
```py
C0 = C1 = 1
Cn = sigma(Ci * Cn-1-i) i=0 -> n
C3 = C0 * C2 + C1 * C1 + C2 * C0
```
15. Create a 2D array:
```py
arr = [[2,3,4] for i in range(2)]
arr = [[3]*n for i in range(3)]
```

16. Sort a dictionary based on its values in descending order `sorted(d, key=d.get, reverse=True)`:
```py
freq # dictionary  
for ele in sorted(freq, key=freq.get, reverse=True):
  print(ele, freq[ele)
```
16.1. `sorted(myDict)` ona dictionary without any key returns an array of dict keys sorted in **ascending order**.


17. `arr.sort()` sorts the list in-place and it's only for lists but `sorted(arr, key=arr.get, reverse=True)` creates a new one and it's used for any iterable like list, dictionary, ...

17.1. To sort an array based on a function like `abs` absolute value of each element: `sorted(arr, key=abs)`. So for an array with pos and neg values: `[2,-4,0,0,-8,1]`, the result of `sorted(arr, key=abs)` will be `[0, 0, 1, 2, -4, -8]`.

17.2. If we have a list and we want to create a dictionary where elements are stored as dictionary keys and their counts are stored as dictionary values, we can use `collections.Counter(arr)`:
```py
arr = collections.Counter([3,1,3,6]) # Counter({3: 2, 1: 1, 6: 1})
```

17.3. for `Counter` we can either use a list or a string, both are the same:
    - `collections.Counter("apple")`
    - `collections.Counter(list("apple"))`

18. `ord()` method converts a character into its Unicode code value. 
    - Try to avoid using `int` and use `ord` in problems when we convert a string to integer because `int` conversion increases time significantly. To convert a string to a integer:
    ```py
    numStr = '4'
    num = ord(numStr) - ord('0')  => that will give num = 4
    ```
    - to first check if a char is digit or not
    ```py
    s[i].isdigit()
    ```
    - `ord()` is useful, if you want to check whether a string contains a character. `ord(char)`. We can also create an array of all characters with their Unicode value as the key and their frequency in a string as a value:
```py
chars = [0]*128
for char in s:
    chars[ord(char)] += 1
```

19. 
    - `s.strip()` : returns **a new string** after removing **any leading and trailing whitespaces including tabs (\t)**
    - `s.lstrip()` : returns **a new string** after removing **any leading and trailing whitespaces from the left of the string**
    - `s.rstrip()` : returns **a new string** after removing **any leading and trailing whitespaces from right of the string **

20. `zip` and `*`: zip takes iterables (zero or more) and aggregates them in a tuple and return it. The zip()function returns an iterator of tuples based on the iterable objects.

    - If we do not pass any parameter, zip() returns an empty iterator
    - If a single iterable is passed, zip() returns an iterator of tuples with each tuple having only one element.
    - If multiple iterables are passed, zip() returns an iterator of tuples with each tuple having elements from all the iterables.

    - Suppose, two iterables are passed to zip(); one iterable containing three and other containing five elements. Then, the returned iterator will contain three tuples. It's because iterator stops when the shortest iterable is exhausted.
```py
number_list = [1, 2, 3]
str_list = ['one', 'two', 'three']

# No iterables are passed
result = zip()

# Converting iterator to list
result_list = list(result)
print(result_list)   #[]

# Two iterables are passed
result = zip(number_list, str_list)

# Converting iterator to set
result_set = set(result)
print(result_set) # {(2, 'two'), (3, 'three'), (1, 'one')}


numbersList = [1, 2, 3]
str_list = ['one', 'two']
numbers_tuple = ('ONE', 'TWO', 'THREE', 'FOUR')

# Notice, the size of numbersList and numbers_tuple is different
result = zip(numbersList, numbers_tuple)

# Converting to set
result_set = set(result)
print(result_set)  # {(2, 'TWO'), (3, 'THREE'), (1, 'ONE')}

result = zip(numbersList, str_list, numbers_tuple)

# Converting to set
result_set = set(result)
print(result_set)  # {(2, 'two', 'TWO'), (1, 'one', 'ONE')}
```

- The * operator can be used in conjunction with zip() to unzip the list. `zip(*zippedList)`:
```py
coordinate = ['x', 'y', 'z']
value = [3, 4, 5]

result = zip(coordinate, value)
result_list = list(result)
print(result_list)   # [('x', 3), ('y', 4), ('z', 5)]

c, v =  zip(*result_list)
print('c =', c)  # c = ('x', 'y', 'z')
print('v =', v)  # v = (3,4,5)
```
21. In a dictionary to check for a key or to get key/vale and also to check if a key/string starts with some value:
```py
dd= {'apple':3, 'app': 5, 'lsd':78}

for key, val in dd.items():
    print(key, val)
    if key.startswith('app'):
        print(key, val)
```
- to get a value of a key from a dictionary and if the key does not exist a value to return:
```py
dd.get(keyName, value)
keyname: the keyname of the item you want to return the value from 
value: optional, to return if the key does not exist.
```

22. For just getting elements that are excluded we can use diff between sets:
```py
orig = set(['1','2','8','7'])
dup = set('1', '8'])

for ele in orig - dup:
    print(ele)  # '2', '7'
```
22. list comprehension, join, split, map:
- by adding if else at the beginning of list comprehension we not only filter but change the value: `print([x if x[0] > 8 else (0,0) for x in arr30])`
- For filtering if cond without else should come at the end: `print([x for x in arr30 if x[0] > 8 ])`
```py
rangestr = '5:10 ,1:7 , 8:10  , 10:15, 13:14'
# [(1,7), (5,10), (8,10),(10,15),(13,14)]

arr1 = rangestr.split(',')
arr20 = [tuple(x.strip().split(':')) for x in arr1]
arr30 = [(int(x[0]), int(x[1])) for x in arr20]
print(arr30.sort(key=lambda x: x[0]))

print([x if x[0] > 8 else (0,0) for x in arr30])


# arr1 = rangestr.split(',')
# arr2 = map(lambda x: x.strip(), arr1)
# arr3 = map(lambda x: tuple(x.split(':')), arr2)
# print(arr3)
# arr4 = map(lambda x: (int(x[0]), int(x[1])), arr3)
# print(arr4)
# arr4.sort(key=lambda x:x[0])
# print(arr4)

# arr5 = ",".join(map(lambda x: str(x[0])+':'+str(x[1]), arr4))
# print(arr5)


arr1 = rangestr.split(',')
arr2 = [x.strip() for x in arr1]
arr3 = [tuple(x.split(':')) for x in arr2]
print(arr3) 
arr4 = [(int(x[0]), int(x[1])) for x in arr3]
arr4.sort(key=lambda x: x[0])
print(arr4)
print(','.join([str(x[0])+':'+str(x[1]) for x in arr4]))
```
- list comprehension is more pytonic, + conditional for filtering or changing an output:
```py

```
23. - **`ss.isalpha(), ss.isnumeric()` to check if its a letter or number**


24. **Python combination and permutation**:
- for a given input if we should not change the place of elements in that input, use combination (n choose k: n!/(n-k)! k!):
```py
cc = combonations('ABC', 2)  # returns an iterable
for ele in cc:
  print(cc)  # returns 2-length tuple in sorted order
  
('A', 'B')
('A', 'C')
('B', 'C')
```
- if we can change the order, user permutation (n permutes k : n!/(n-k)!):
```py
cc = permutations('ABC', 2)  # returns an iterable
for ele in cc:
  print(cc)  # returns 2-length tuples all possible ordering
  
('A', 'B')
('A', 'C')
('B', 'A')
('B', 'C')
('C', 'A')
('C', 'B')
```

25. **PriorityQueue**
- It is a Heap data structure
- `from Queue import PriorityQueu`
- The lowest valued entries are retrieved first. A typical pattern for entries is a tuple in the form: `(priority_number, data)`:
```py
from Queue import PriorityQueue

q = PriorityQueue()

""" 
i is a counter for when the comparison between values is equal and the data is not
comparable so we add i to be compared.
"""

q.put((3, i, 'Read')
q.put((5, i, "Write'))

OR 

q.put(4)
############

q.empty() # check if it's empty

############

val = q.get()  // remove and return the item
val, nn = q.get()
```

26. **Heap**
- Heap is a binray tree with 2 characteristics:
    1. Heaps must be Complete binary trees
    2. The nodes must be ordered according to the Heap order property. Two heap order properties are as follows:
        - **Max Heap**: root node has the max value (parent node >= children node) 
        - **Min Heap**: root has the min value (parent node <= children node)
    3. In python we only have min Heap (smallest number at the top of the heap).
```py
from heapq import *

ll = [1,2,3]
heapify(ll) # to transform a populated list into a heap in-place

ll = [(-freq, val) for key, val in c.items()] # heapify based on -freq value (min-heap)
heapq.heapify(ll)


heappush(arr, val)
heappop(arr)

```
- If the value we are pushing into heap is not comparable, we can also pass in a tuple to heap instead:
```py
heappush(myHeap, (priority_value, data) # heappush(myHeap, (4, 'helloworld)

#for comparing the linked nodes, it will error out because it doesn't know how to compare the node. We also need to add these to the definition of list node otherwise it errors out as it can't compare. if that doesn't work use PriorityQueue instead:

class Node:
    def __init__(self, val=None):
        self.val = val
        self.next = None
        
    def __eq__(self, other):
        return self.val == other.val
    
    def __ne__(self, other):
        return not (self.val == other.val)
    
    def __gt__(self, other):
        return self.val > other.val

    def __lt__(self, other):
        return self.val < other.val

    def __le__(self, other):
        return self.val <= other.val

    def __ge__(self, other):
        return self.val >= other.val
```

27. `someStr.zfill(width)` returns a copy of the string **left filled with ASCII `0`** digits to make a string of lenght `width`.
```py
>>> "42".zfill(5)
'00042'
>>> "-42".zfill(5)
'-0042'
```

28. Reversing a list: `ll = myList[::-1]`

####################################################################################
- In dividing a even or odd array into two partitions and later on calculate the median. We put the partition line in a place where the left side partition would always have one more element than the right side so the median was calculated:
    - if the array has even number of elements: median = (left of partition + partition line) / 2
    - if the array has odd number of elements: median = left of partition
Then we need to divide the array this way: `(totalSize + 1)/2`. 


- When trying to add two numbers (defined in reverse by linked lists), we can always start from least-significant digit and have a `carry` variable to keep track of carry. Don't forget to add the carry at the end, if it is 1 (not 0).

- In an array if we try to find for each x, whether `2x` or `x/2` exits, to simplify, we can first order the array based on absolute value while maintaining the signs by using `sorted(arr, key=abs)` to start from the least value x and then only check for the existence of 2x. For that we also need to create a hash map with key==numbers in array and value==frequency of numbers `collections.Counter(arr)` so that everytime, we find 2x for a x, we remove both them (-1 frequency) from the hash map, so they won't be used more than once.

- Given an integer array of even length arr, return true if it is possible to reorder arr such that arr[2 * i + 1] = 2 * arr[2 * i] for every 0 <= i < len(arr) / 2, or false otherwise. The elemnts can be pos or negative and repeating:

- For stone game where there is an array with piles of stones as its elements, two playes play and each time they can only get stones from the left or right of the array. Determine which player wins?
    - We can think of DP and think of a helper function that takes the first and last indices of the array (recursion) and returns a tuple of `(total stones for the first player, total stones of the second player)`. Now when the first player go, they can either take the left elem + the second choice of the remaining array or take the right ele + the second choice of the remaining array. (Second choice cause the second player will take the first choice). We then try to optimize it by taking the max value. Since the choices are repeated, we can use memoization to save in computation of the value for the same orders. (https://leetcode.com/submissions/detail/537560546/)

### Some data structure patterns (https://betterprogramming.pub/the-ultimate-strategy-to-preparing-for-the-coding-interview-ee9f7eb439f3)
1. If the given input is sorted (array, list, or matrix), we will use a variation of **Binary Search** or a **Two Pointers** strategy.
2. If we’re dealing with **top/maximum/minimum/closest k elements among n elements**, we will use a **Heap**.
3. If we **need to try all combinations (or permutations) of the input**, we can either use **recursive Backtracking** or **iterative Breadth-First Search**.

1.1. Find a pair of numbers with sum equal to target in a sorted array:
- we can use two pointers approach to go through the elements (starting with the first and last element in the array) and adjust the pointers based on their sum compared with the target value:
```py
def targetSum(nums, target):
    left, right = 0, len(nums) - 1
    
    while left < right:
        sumVal = nums[left] + nums[right]
        
        if sumVal == target:
            return (left, right)
        elif sumVal > target:
            right = right - 1
        else:
            left = left + 1

```
- You were recently handed a comma-separated value (CSV) file that was horribly formatted. Your job is to extract each row into an list, with each element of that list representing the columns of that file. What makes it badly formatted? The “address” field includes multiple commas but needs to be represented in the list as a single element! Assume that your file has been loaded into memory as the following multiline string:
```csv
Name,Phone,Address
Mike Smith,15554218841,123 Nice St, Roy, NM, USA
Anita Hernandez,15557789941,425 Sunny St, New York, NY, USA
Guido van Rossum,315558730,Science Park 123, 1098 XG Amsterdam, NL
```
your output should be:
```py
[
    ['Mike Smith', '15554218841', '123 Nice St, Roy, NM, USA'],
    ['Anita Hernandez', '15557789941', '425 Sunny St, New York, NY, USA'],
    ['Guido van Rossum', '315558730', 'Science Park 123, 1098 XG Amsterdam, NL']
]
```
Ans:

```py
input_string = """Name,Phone,Address
Mike Smith,15554218841,123 Nice St, Roy, NM, USA
Anita Hernandez,15557789941,425 Sunny St, New York, NY, USA
Guido van Rossum,315558730,Science Park 123, 1098 XG Amsterdam, NL"""


def stringSplit(unsplit):
    results = []
    
    lines = unsplit.splitlines()
    
    for line in lines[1:]:
        print('line: ', line)
        results.append(line.split(',', maxsplit=2))
        
    return results

print(stringSplit(input_string))
        ```

- Using our web scraping tutorial, you’ve built a great weather scraper. However, it loads string information in a list of lists, each holding a unique row of information you want to write out to a CSV file:
```py
[
    ['Boston', 'MA', '76F', '65% Precip', '0.15 in'],
    ['San Francisco', 'CA', '62F', '20% Precip', '0.00 in'],
    ['Washington', 'DC', '82F', '80% Precip', '0.19 in'],
    ['Miami', 'FL', '79F', '50% Precip', '0.70 in']
]
```
Your output should be a single string that looks like this:
```py
"""
Boston,MA,76F,65% Precip,0.15in
San Francisco,CA,62F,20% Precip,0.00 in
Washington,DC,82F,80% Precip,0.19 in
Miami,FL,79F,50% Precip,0.70 in
"""
```
- Ans:
```py
def joinList(data):
    joined = [','.join(line) for line in data]
    output = '\n'.join(joined)
    return output
```
