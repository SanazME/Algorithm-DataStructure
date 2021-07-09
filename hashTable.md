## Hash Tables
- Hash Table is a data structure which organizes data using **hash functions** in order to support quick insertion and search.
- The key idea of Hash Table is to use a hash function to **map keys to buckets**. To be more specific,
  - When we insert a new key, the hash function will decide which bucket the key should be assigned and the key will be stored in the corresponding bucket;
  - When we want to search for a key, the hash table will use the **same hash function** to find the corresponding bucket and search only in the specific bucket.

## Keys to Design a Hash Table
There are two essential factors that you should pay attention to when you are going to design a hash table.
**1. Hash Function**
**2. Collision Resolution**

### 1. Hash Functions
The hash function is the most important component of a hash table which is used to map the key to a specific bucket. In the example in previous article, we use y = x % 5 as a hash function, where x is the key value and y is the index of the assigned bucket.

The hash function will depend on the **range of key values** and the **number of buckets.** The idea is to try to **assign the key to the bucket as uniform as you can**. Ideally, a perfect hash function will be a one-one mapping between the key and the bucket. However, in most cases a hash function is not perfect and it is a **tradeoff between the amount of buckets and the capacity of a bucket.**

## 2. Collision Resolution
Ideally, if our hash function is a perfect one-one mapping, we will not need to handle collisions. Unfortunately, in most cases, collisions are almost inevitable. For instance, in our previous hash function (y = x % 5), both 1987 and 2 are assigned to bucket 2. That is a collision.

A collision resolution algorithm should solve the following questions:

1 How to organize the values in the same bucket?

2. What if too many values are assigned to the same bucket?

4. How to search a target value in a specific bucket?

These questions are related to the capacity of the bucket and the number of keys which might be mapped into the same bucket according to our hash function.

Let's assume that the **bucket**, which holds the **maximum number of keys, has N keys**.

Typically, if **N is constant and small**, we can simply use an **array to store keys in the same bucket**. If **N is variable or large**, we might need to use **height-balanced binary search tree** instead.

### Space and time complexity
- If there are M keys in total, we can achieve the space complexity of O(M) easily when using a hash table.
- In built-in hash tables: The average time complexity of both **insertion and search** is still **O(1)**. And the time complexity in the **worst case is O(logN) for both insertion and search by using height-balanced BST**. It is a trade-off between insertion and search.
- 
- Most of us might have used an array in each bucket to store values in the same bucket. Ideally, the bucket size is small enough to be regarded as a constant. The time complexity of both insertion and search will be O(1). But in the worst case, the maximum bucket size will be N. And the time complexity will be O(1) for insertion but O(N) for search.


**The typical design of built-in hash table is:**

- The key value can be any **hashable type**. And a value which belongs to a hashable type will have a **hashcode**. This code will be used in the mapping function to **get the bucket index.**
- **Each bucket contains an array to store all the values in the same bucket initially.**
- If there are too many values in the same bucket, these values will be maintained in a **height-balanced binary search tree instead.**


## Sliding Window
- when you see s substring problem: "longest substring with a condition", think about sliding window pattern: [https://leetcode.com/problems/longest-substring-without-repeating-characters/](https://leetcode.com/problems/longest-substring-without-repeating-characters/)
- Time complexity O(n)
- Space complexity: hashmap O(min(m,n)). m is the size of charset
- Space complexity table O(m)

```py
def lengthOfLongestSubstring( s):
    """
    :type s: str
    :rtype: int
    """
    if len(s) <= 1:
        return len(s)

    start = 0
    maxLen = 0
    seen = {}
    
    for i, char in enumerate(s):
        print('char, i: ', char, i)
        print('seen: ', seen)
        print('start: ', start)
        if char in seen and start <= seen[char]:
            start = seen[char] + 1
            seen[char] = i
        else:
            # print(start)
            maxLen = max(maxLen, i - start + 1)
            seen[char] = i
    return maxLen
        
```
