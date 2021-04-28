# Algorithm-DataStructure

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
hashset.remove(3)
hashset.clear() # ()
```
7. To find a number in an array that does not repeated twice (all other numbers do) and to not use extra space, we can use XOR (exclusive OR) operation:
a XOR a = 0
a XOR 0 = a
XOR is associative
so if we XOR all the numbers the non repeated number will be returned: 1 XOR 4 XOR 1 XOR 2 XOR 2 = (1 XOR 1 ) XOR 4 XOR (2 XOR 2) = 0 XOR 4 = 4
