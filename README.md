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

**Serialization and Deserialization**
- Serialization is the process of converting a data structure or object into a sequence of bits so that it can be saved in a file or memory buffer, or transmitted across a network connection.

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
17. `arr.sort()` sorts the list in-place and it's only for lists but `sorted(arr, key=arr.get, reverse=True)` creates a new one and it's used for any iterable like list, dictionary, ...

17.1. To sort an array based on a function like `abs` absolute value of each element: `sorted(arr, key=abs)`

17.2. If we have a list and we want to create a dictionary where elements are stored as dictionary keys and their counts are stored as dictionary values, we can use `collections.Counter(arr)`:
```py
arr = collections.Counter([3,1,3,6]) # Counter({3: 2, 1: 1, 6: 1})
```

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
