## Write a collection of data structure that is fast (should run O(1) time amortized) for each of the following operation - add(element), remove(ele), get(ele).
- Insert Delete GetRandom O(1): https://leetcode.com/problems/insert-delete-getrandom-o1/
- If we have to implement LRU cache approach, we would use a hash table and DB linked list. The hash table keeps the key and linked list nodes so we can find a node in O(1) time. We can insert the node always after header node so it'll be O(1) time. For removing a node, we find the node in O(1) from the hash table and then we can just remove it in O(1). However, to implement LRU cache, we want to always move the node after the header if we add it or if we make a get query and we want remove the leaset used node if we reach the capacity by moving the last node before the tail node.

- IF we don't implement LRU cache and we just O(1) in all those operations, we can either use a hash table or an **undorted array**: https://en.wikipedia.org/wiki/Search_data_structure.  Insert on an unsorted array is sometimes quoted as being O(n) due to the assumption that the element to be inserted must be inserted at one particular location of the array, which would require shifting all the subsequent elements by one position. However, in a classic array, the array is used to store arbitrary unsorted elements, and hence the exact position of any given element is of no consequence, and insert is carried out by increasing the array size by 1 and storing the element at the end of the array, which is a O(1) operation.Likewise, the deletion operation is sometimes quoted as being O(n) due to the assumption that subsequent elements must be shifted, but in a classic unsorted array the order is unimportant (though elements are implicitly ordered by insert-time), so deletion can be carried out by swapping the element to be deleted with the last element in the array and then decrementing the array size by 1, which is a O(1) operation.

- Now with implementing a getRandom function, we can't only use hashtable or an array. Hashmap provides Insert and Delete in average constant time, although has problems with GetRandom. The idea of GetRandom is to choose a random index and then to retrieve an element with that index. There is no indexes in hashmap, and hence to get true random value, one has first to convert hashmap keys in a list, that would take linear time. The solution here is to build a list of keys aside and to use this list to compute GetRandom in constant time. Array has the issues we mentioned earlier.

- The solution is have a hashtable and an array. Hashtable to store the elements as key and their index as value so we know where they are in the array. To delete a value at arbitrary index takes linear time. The solution here is to always delete the last value:

    - Swap the element to delete with the last one.

    - Pop the last element out.

```py
class RandomizedSet(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.hashTable = {}
        self.arr = []
        self.idx = 0
                
        
    def insert(self, val):
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        :type val: int
        :rtype: bool
        """
        if val in self.hashTable:
            return False
        
        # insert in array
        self.arr.append(val)
                
        # insert in hash table
        self.hashTable[val] = self.idx
        
        # increment index 
        self.idx += 1
        
        return True
        
    def remove(self, val):
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        :type val: int
        :rtype: bool
        """
        if val not in self.hashTable:
            return False
        
        loc = self.hashTable[val]
        
        # if the val is NOT at the end of array
        if loc != self.idx:
            self._swapWithLast(loc)

            
        self.arr.pop()
        self.idx -= 1
        del self.hashTable[val]
        

        return True
        
        
    def _swapWithLast(self, loc):
        # swap with the last element
        self.arr[-1], self.arr[loc] = self.arr[loc], self.arr[-1]
        
        # update hash table value (location of last element)
        self.hashTable[self.arr[loc]] = loc

    def getRandom(self):
        """
        Get a random element from the set.
        :rtype: int
        """
        return random.choice(self.arr)
        

# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()

```

## Given a list of leaf nodes, return a list containing all the nodes of the tree in DFS order.
- https://leetcode.com/discuss/interview-question/859662/Twitter-Phone-Interview
- Given a list of leaf nodes, return a list containing all the nodes of the tree in DFS order.
'''
class Node{
int id;
List parentIds; //this will be in the descending order in terms of the precedence of the parents.
}
'''
Example: Given a tree
    1
  2  4
 /   \
3    5

Input List -{3,5}
o/p: {1,2,3,4,5}

```py
class Node(object):
    def __init__(self, idVal, parentIds):
        self.id = idVal
        self.parentIds = parentIds
        

leafNode1 = Node(5, [4,2,1])
leafNode2 = Node(6, [2,1])
leafNode3 = Node(9, [7,1])
leafNode4 = Node(8, [7,1])

        
def dfs(leafNodes):
    result = []
    visited = set()
    for leafNode in leafNodes:
        parents = leafNode.parentIds
        for i in range(len(parents)-1, -1, -1):
            if parents[i] not in visited:
                visited.add(parents[i])
                result.append(parents[i])
        
        if leafNode.id not in visited:
            result.append(leafNode.id)
        print('result: ', result)
        
    return result
      
leafNodes = [leafNode1, leafNode2, leafNode3, leafNode4]

print(dfs(leafNodes))

```
## Tweet Counts Per Frequency
- https://leetcode.com/problems/tweet-counts-per-frequency/
- A social media company is trying to monitor activity on their site by analyzing the number of tweets that occur in select periods of time. These periods can be partitioned into smaller time chunks based on a certain frequency (every minute, hour, or day).

For example, the period [10, 10000] (in seconds) would be partitioned into the following time chunks with these frequencies:
    - Every minute (60-second chunks): [10,69], [70,129], [130,189], ..., [9970,10000]
    - Every hour (3600-second chunks): [10,3609], [3610,7209], [7210,10000]
    - Every day (86400-second chunks): [10,10000]
Notice that the last chunk may be shorter than the specified frequency's chunk size and will always end with the end time of the period (10000 in the above example).

Design and implement an API to help the company with their analysis.

Implement the TweetCounts class:

TweetCounts() Initializes the TweetCounts object.
void recordTweet(String tweetName, int time) Stores the tweetName at the recorded time (in seconds).
List<Integer> getTweetCountsPerFrequency(String freq, String tweetName, int startTime, int endTime) Returns a list of integers representing the number of tweets with tweetName in each time chunk for the given period of time [startTime, endTime] (in seconds) and frequency freq.
freq is one of "minute", "hour", or "day" representing a frequency of every minute, hour, or day respectively.
    
- **One important to note that one tweet can be tweeted multiple times and so it can have more than one timestamp**.
- To find the number of intervals based on startTime, endTime and freq value: 
```py
q, r = divmod((endTime - startTime + 1), freqVal) 
if r != 0:
    intervals = q + 1
else:
    intervals = q
```

```py
 class TweetCounts(object):
    def __init__(self):
        self.tweetTable = {}
        self.freqMap = {
            'minute': 60,
            'hour': 60 * 60,
            'day': 24 * 3600
        }
        
    def recordTweet(self, tweetName, time):
        """
        tweetName: string
        time: int
        save tweetName and time
        """
        if tweetName in self.tweetTable.keys():
            self.tweetTable[tweetName].append(time)
        else:
            self.tweetTable[tweetName] = [time]
        return True
        
     
    def _createTimeChunks(self, freq, startTime, endTime):
        """
        freq: string (minute, hour, day)
        startTime: int
        endTime: int
        
        Return int : number of time chunks for a given freq and start and end times
        """
        
        chunkCount = ((endTime - startTime + 1) // self.freqMap[freq])
        return chunkCount
        
    def getTweetCountsPerFrequency(self, freq, tweetName, startTime, endTime):
        """
        freq: string (minute, hour, day)
        tweetName: string
        startTime: int
        endTime: int
        
        Return a list of ints [int] representing number of tweets with tweetName in each time chunk within start and end time period.
        """
        freqChunks = self._createTimeChunks(freq, startTime, endTime)
        result = [0] * freqChunks
                
        for t in self.tweetTable[tweetName]:
            if (t >= startTime) and (t <= endTime):
                idx = (t - startTime)//self.freqMap[freq]
                result[idx] += 1
                
        return result  
```
## Design Log Storage System
You are given several logs, where each log contains a unique ID and timestamp. Timestamp is a string that has the following format: `Year:Month:Day:Hour:Minute:Second`, for example, `2017:01:01:23:59:59`. All domains are zero-padded decimal numbers.

Implement the LogSystem class:

- `LogSystem()` Initializes the LogSystem object.
- `void put(int id, string timestamp)` Stores the given log (id, timestamp) in your storage system.
- `int[] retrieve(string start, string end, string granularity)` Returns the IDs of the logs whose timestamps are within the range from start to end inclusive. start and end all have the same format as timestamp, and granularity means how precise the range should be (i.e. to the exact Day, Minute, etc.). For example, start = "2017:01:01:23:59:59", end = "2017:01:02:23:59:59", and granularity = "Day" means that we need to find the logs within the inclusive range from Jan. 1st 2017 to Jan. 2nd 2017, and the Hour, Minute, and Second for each log entry can be ignored.
- Let's focus on the retrieve function. For each granularity, we should consider all timestamps to be truncated to that granularity. For example, if the granularity is 'Day', we should truncate the timestamp '2017:07:02:08:30:12' to be '2017:07:02'. Now for each log, if the truncated timetuple cur is between start and end, then we should add the id of that log into our answer.
                
- **the reason that we use tuple and not list to split the timestamp from `:` and save is that list is mutable and so when we modify the time based on granularity, we corrupt the data. If we use tuple, it is immutable and the original data will be intact.**
                
- **Note that a list of string or tuple of strings can be compared like : `print(['2000', '03', '23', '20', '47', '37'] < ['2001', '00', '00', '00', '00', '00'])` return True**
- we can store logs and timestamps in a dic with `key = tuple(timestamp.split(":"))`. When retrieving data, we first convert the start and end time to the granular level we want (meaning the rest part of time will be 00) and then we compare the `timestamp[:idx]` tuple with startGran and endGran tuples up to the granular idx: `startGran[:idx]...`

```py
class LogSystem(object):

    def __init__(self):
        self.logs = {}
        self.granularityMap = {
            'Year': 1,
            'Month': 2, 
            'Day': 3,
            'Hour': 4,
            'Minute': 5,
            'Second': 6
        }
        

    def put(self, id, timestamp):
        """
        :type id: int
        :type timestamp: str
        :rtype: None
        """
        key = tuple(timestamp.split(':'))
        self.logs[key] = id
        
    def _makeGranular(self, time, granularity):
        timeGran = ['1999', '00', '00', '00', '00', '00']
        timeList = time.split(':')
        
        for i in range(self.granularityMap[granularity]):
            timeGran[i] = timeList[i]
        
        return tuple(timeGran)
        

    def retrieve(self, start, end, granularity):
        """
        :type start: str
        :type end: str
        :type granularity: str
        :rtype: List[int]
        """
        startGran = self._makeGranular(start, granularity)
        endGran = self._makeGranular(end, granularity)
        idx = self.granularityMap[granularity]
        ids = []
        
        for timestamp, id in self.logs.items():
            if (timestamp[:idx] >= startGran[:idx]) and (timestamp[:idx] <= endGran[:idx]):
                ids.append(id)
                
        return ids
        


# Your LogSystem object will be instantiated and called as such:
# obj = LogSystem()
# obj.put(id,timestamp)
# param_2 = obj.retrieve(start,end,granularity)
```
## The kth Factor of n
- https://leetcode.com/problems/the-kth-factor-of-n/
- One way O(n) to march from 1 till n and see if you find a dvisor and if so increament a counter till it reached the value of k and then return that element or at the end return -1:
```py
def kthFactor(self, n, k):
    """
    :type n: int
    :type k: int
    :rtype: int
    """
    count = 0
    for num in range(1, n+1):
        if n % num == 0:
            count += 1
        if count == k:
            return num

    return -1                
```
- One improvement, we know that if i is a divisor of n then n/i is also a divisor of n. So instead of iterating all the way to n, we need to iterate till `sqrt(n)` but then we need to push those two divisors in a max heap so we can always access the largest divisor. SO we can use heap to store the numbers with the max at the top (python heap is min heap so to keep max element always on top we need to push negative values.
```py
import heapq

def kthFactor(self, n, k):
    """
    :type n: int
    :type k: int
    :rtype: int
    """
    #push into heap
    #by limiting size of heap to k
    def heappush_k(num):
        heappush(heap, -num)
        if len(heap) > k:
            heappop(heap)

    # Python heap is min heap 
    # -> to keep max element always on top,
    # one has to push negative values
    heap = []
    for num in range(1, int(sqrt(n) + 1)):
        if n % num == 0:
            heappush_k(num)
            if num != n // num:
                heappush_k(n // num)

    if k == len(heap):
        res = - heappop(heap)
    else:
        res = -1

    return res
        
```
## Maximum Number of Occurrences of a Substring
- https://leetcode.com/problems/maximum-number-of-occurrences-of-a-substring/
- Hints: The max length of substring can get at max 26 character lenght & since we're looking for max occurance of that substring, we need to find ones with minSize. If a substring with `maxSize` reoccurs, it's garanteed that smaller substring reoccurs at least at the same frequency.
- we use sliding window to move along the string and test the condition for each substring of size `minSize`, i.e, whether its number of unique chars <= `maxLetters`. 
```py
def maxFreq(self, s, maxLetters, minSize, maxSize):
        """
        :type s: str
        :type maxLetters: int
        :type minSize: int
        :type maxSize: int
        :rtype: int
        """
        def uniqueChar(substr):
            return len(set(substr))
        
        def insertFreq(substr, dic):
            if not substr in dic:
                dic[substr] = 1
            else:
                dic[substr] += 1
            return dic[substr]
        
        if len(s) < minSize: return 0
        
        p1 = 0
        p2 = minSize
        maxFreq = 0
        wordFreq = {}
        
        while (p1 + minSize) <= len(s):
            substr = s[p1:p1+minSize]
            count = uniqueChar(substr)
            if count <= maxLetters:
                # insert in dic
                freq = insertFreq(substr, wordFreq) 
                maxFreq = max(maxFreq, freq)
            
            p1 += 1
                
        return maxFreq
```
## Minimum Swaps to Group All 1's Together
- https://leetcode.com/problems/minimum-swaps-to-group-all-1s-together/
- The total number of 1s shows the size of the subarray. If that number is equal to the length of data array or is one, the swap is not needed (0). Then for that subarray, we find the number of zeros and we can have a sliding window to keep track of zeros coming into the subarray and zeros that leaving the subarray.
```py
 def minSwaps(self, data):
    """
    :type data: List[int]
    :rtype: int
    """
    # find number of 1s in arr === size of the subarray with all 1s
    ones = sum(data)

    if len(data) == ones or ones == 1: return 0

    subarr = data[0: ones]
    zeros = 0

    for ele in subarr:
        if ele == 0:
            zeros += 1

    minSwap = zeros

    for i in range(ones, len(data)):
        if data[i] == 0:
            zeros += 1
        if data[i-ones] == 0:
            zeros -= 1

        minSwap = min(minSwap, zeros)

    return minSwap                                                 
```
- another way to solve is to use a queue to append and pop elments and keep track of number of zeros, be aware of edge cases:
```py
 def minSwaps(self, data):
    """
    :type data: List[int]
    :rtype: int
    """
    # find number of 1s in arr === size of the subarray with all 1s
    ones = sum(data)

    if len(data) == ones or ones == 1 or ones == 0: return 0

    queue = collections.deque()
    zeros = 0
    minZeros = float('Inf')

    for i in range(len(data)):

        if len(queue) < ones:
            queue.append(data[i])
            if data[i] == 0:
                zeros += 1

        else:
            minZeros = min(minZeros, zeros)
            ele = queue.popleft()
            if ele == 0: 
                zeros -= 1
            queue.append(data[i])
            if data[i] == 0:
                zeros += 1
            minZeros = min(minZeros, zeros)

    return minZeros                                               
```
## Maximal Square
- https://leetcode.com/problems/maximal-square/
- explaination: https://www.youtube.com/watch?v=RElcqtFYTm0
- with DP, we can solve for the current state if we know the prev state (solution). Here from bottom-top approach, we can create a 2D array for dp and starting from top-left, first we populate the first row and col with the exact values. For each location dp(i,j), we look at the 3 prev direction and take the min of all + 1. If the matrix(i,j) is 0 we just set dp as 0.
- dp(i,j) represents the side length of the maximum square whose bottom right corner is the cell with index (i,j) in the original matrix. Starting from index (0,0), for every 1 found in the original matrix, we update the value of the current element as

dp(i, j) = min(dp(i-1, j), dp(i-1, j-1), dp(i, j-1)) + 1
```py
def maximalSquare(self, matrix):
    """
    :type matrix: List[List[str]]
    :rtype: int
    """
    rows = len(matrix)
    cols = len(matrix[0])
    maxSoFar = 0

    dp = [[0 for j in range(cols)] for i in range(rows)]

    # populate the first row and col in dp with values from matrix
    for i in range(rows):
        dp[i][0] = int(matrix[i][0])
        maxSoFar = max(maxSoFar, dp[i][0])

    for j in range(cols):
        dp[0][j] = int(matrix[0][j])
        maxSoFar = max(maxSoFar, dp[0][j])

    for i in range(1, rows):
        for j in range(1, cols):
            if matrix[i][j] == "1":
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
            else:
                dp[i][j] = 0

            maxSoFar = max(maxSoFar, dp[i][j])
    print(dp)

    return maxSoFar*maxSoFar
                  
```
## Find a median in a stream of integers
- condition 1: what if the numbers are only between 1 and 100?
- what if the range of numbers are really high?
    
    
    
    
## Find elements in a pair list which also contain the reverse pair?

    
## Twitter Stickers 
- We have a bunch of stickers which say "twitter" and we decide to cut these up into their separate letters to make new words. So, for example, one sticker would give us the letters "T", "W", "I", "T", "T", "E", "R", which we could rearrange into other word[s] (like "write", "wit", "twit", etc)

Challenge:
Write a function that takes as its input an arbitrary string and as output, returns the number of intact “twitter” stickers we would need to cut up to recreate that string.

Example: twitter_stickers(“write wit twit”) would return "3", since we would need to cut up 3 stickers to provide enough letters to write “write wit twit”
```py
import collections
def twitter_stickers(s):
    stickerHash = collections.Counter('twitter')
    wordFreq = collections.Counter("".join(s.split(' ')))
    print('wordFreq: ', wordFreq)
    
    maxCount = 0
    for key in wordFreq:
        if key not in stickerHash:
            return 0
        maxCount = max(wordFreq[key] - stickerHash[key], maxCount)
        
    return maxCount + 1
    
    
    
# print(twitter_stickers("write wit twit"))
# print(twitter_stickers("twitter"))
# print(twitter_stickers("rty"))
# print(twitter_stickers("ttttww"))
```
## Add Likes
- For the first feature of the Twitter application, we are creating an API that calculates the total number of likes on a person’s Tweets. For this purpose, your team member has already extracted the data and stored it in a simple text file for you. You have to create a module that takes two numbers at a time and returns the sum of the numbers. The inputs were extracted from a text file, so they are in the form of strings. The limitation of the API is that we want all of the values to remain strings. We cannot even convert the string to numbers temporarily.

For example, let’s say you are given the strings "1545" and "67" as input. Your module should not convert these into integers while computing the sum and return "1612".

**Time complexity**: `O(max(N, M))`
**Space complexity**: `O(max(N, M))`
    
```py
 def add_likes(like1, like2):
    
    # if one of like strings is empty
    if not (like1 and like2):
        return like1 if like1 else like2
    
    n1, n2 = len(like1), len(like2)
    p1, p2 = n1 - 1, n2 - 1
    
    res = ''
    carry = 0
    
    while p1 >= 0 or p2 >= 0:
        
        if p1 >= 0:
            num1 = ord(like1[p1]) - ord('0')
        else:
            num1 = 0
            
        if p2 >= 0:
            num2 = ord(like2[p2]) - ord('0')
        else:
            num2 = 0
        
        sumTwo = num1 + num2 + carry
        res = str(sumTwo % 10) + res
        carry = (sumTwo // 10)
        print(res, carry)
        
        p1 -= 1
        p2 -= 1
        
    if carry > 0:
        res = str(carry) + res
        
    return res
        
     
print(add_likes("1545", "67"))   
```
