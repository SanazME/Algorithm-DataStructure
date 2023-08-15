## 0. Longest Substring Without Repeating Characters
- https://leetcode.com/problems/longest-substring-without-repeating-characters/description
```py
class Solution:
    def lengthOfLongestSubstring(self, s):
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
            if char in seen and start <= seen[char] :
                start = seen[char] + 1
                seen[char] = i
            else:
                maxLen = max(maxLen, i - start + 1)
                seen[char] = i
        return maxLen

```

## 1. Longest Palindromic Substring
- https://leetcode.com/problems/longest-palindromic-substring/
```py
def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """           
        if len(s) <= 1:
            return s
    
        soFar = ''
        for i, char in enumerate(s):
            # for odd case: 'aba'
            tmp = self.helper(s, i, i)
            if len(tmp) >= len(soFar):
                soFar = tmp

            # for even case: 'abba'
            tmp = self.helper(s, i, i+1)

            if len(tmp) >= len(soFar):
                soFar = tmp

        return soFar
        
        
  def helper(self, s, left, right):
      while left >= 0 and right < len(s) and s[left] == s[right]:
          left -= 1
          right += 1
      return s[left+1:right]
```
To check if a word is palindrom coming from out to in:
```py
def isPalindrome(s, start, end):
    if start > end:
        return True
        
    return s[start] == s[end] and isPalindrome(s, start + 1, end - 1)
```


## 2. Web Crawler Multithread - Databrick
- https://leetcode.com/problems/web-crawler-multithreaded/description/

```py
# """
# This is HtmlParser's API interface.
# You should not implement it, or speculate about its implementation
# """
#class HtmlParser(object):
#    def getUrls(self, url):
#        """
#        :type url: str
#        :rtype List[str]
#        """
from concurrent import futures
class Solution:
    def crawl(self, startUrl: str, htmlParser: 'HtmlParser') -> List[str]:
        self.hostname = self.extractHostname(startUrl)


        # paramlist = parameters.split("/")

        visited = set()
        visited.add(startUrl)

        # Solution 1
        # queue = deque([])

        # result = [startUrl]

        # with futures.ThreadPoolExecutor(max_workers=8) as executor:
        #     queue.append(executor.submit(htmlParser.getUrls, startUrl))

        #     while queue:
        #         for url in queue.popleft().result():
        #             print("url: ", url)
        #             if self.extractHostname(url) == hostname and url not in visited:
        #                 result.append(url)
        #                 visited.add(url)
        #                 queue.append(executor.submit(htmlParser.getUrls, url))


        # return result


        # Solution 2
        self.htmlParser = htmlParser
        self.pending = []
        self.seen = set()
        self.count = 0

        with futures.ThreadPoolExecutor(max_workers=8) as self.executor:
            self.submitToExecute(startUrl)
        

            while self.pending:
                self.count += 1
                print("count: ", self.count)
                
                pendingSoFar, self.pending = self.pending, []
                
                for ele in futures.as_completed(pendingSoFar):
                    if e := ele.exception():
                        print(e)
                        

        
        return self.seen


    def submitToExecute(self, url):
        self.seen.add(url)

        print("seen: ", self.seen)

        self.pending.append(self.executor.submit(self.processUrl, url))
        
        


    def processUrl(self, url):
        print(self.extractHostname(url), self.hostname)
        urls = set()
        for uul in self.htmlParser.getUrls(url):
            print("url here: ", uul)
            if self.extractHostname(uul) == self.hostname:
                urls.add(uul)
        # urls = set(url for urls in self.htmlParser.getUrls(url) if self.extractHostname(url) == self.hostname)
        # print("urls: ", urls)
        for url in urls - self.seen:
            self.submitToExecute(url)






    def extractHostname(sefl, url):
        if url.startswith("http://"):
            url = url[7:]
        elif url.startswith("https://"):
            url = url[8:]

        end = url.find("/")

        if end == -1:
            end = len(url)

        else:
            port  = url.find(":")
            if port != -1 and port < end:
                end = port

        hostname = url[:end]
        parameters = url[end:]

        return hostname
```

```java
/**
 * // This is the HtmlParser's API interface.
 * // You should not implement it, or speculate about its implementation
 * interface HtmlParser {
 *     public List<String> getUrls(String url) {}
 * }
 */
class Solution {
    public List<String> crawl(String startUrl, HtmlParser htmlParser) {
        String hostname = getHostname(startUrl);

        List<String> result = new ArrayList<>();
        Set<String> visited = new HashSet<>();

        BlockingQueue<String> queue = new LinkedBlockingQueue<>();
        Deque<Future> tasks = new ArrayDeque<>();

        queue.offer(startUrl);

        ExecutorService executor = Executors.newFixedThreadPool(4, r -> {
            Thread t = new Thread(r);
            t.setDaemon(true);
            return t;
        });

        while (true){
            String url = queue.poll();
            if (url != null) {
                if (getHostname(url).equals(hostname) && !visited.contains(url)){
                    result.add(url);
                    visited.add(url);

                    tasks.add(executor.submit(() -> {
                        List<String> newUrls = htmlParser.getUrls(url);
                        for (String newUrl: newUrls){
                            queue.offer(newUrl);
                        }
                    }
                    ));

                }
            } else {
                if (!tasks.isEmpty()){
                    Future nextTask = tasks.poll();

                    try {
                        nextTask.get();
                    } catch (InterruptedException | ExecutionException e) {}

                } else {
                    break;
                }
            }
        }

        return result;
        
    }



    private String getHostname(String url){
        if (url.startsWith("https://")){
            return url.substring(8).split("/")[0];
        } else if (url.startsWith("http://")){
            return url.substring(7).split("/")[0];
        }
        return url.substring(7).split("/")[0];
    }
}
```

## 3. Container with most water
- https://leetcode.com/problems/container-with-most-water/description/
- start with two pointer from left and right and always move the pointer with the shortest height:
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


## 4. Trapped rain water
- https://github.com/SanazME/Algorithm-DataStructure/blob/master/AmazonQuestions.md#6-trapping-rain-water

## 5. Letter Comibnations of a Phone Number
- https://github.com/SanazME/Algorithm-DataStructure/blob/master/Netflix.md#feature--9-created-all-the-possible-viewing-orders-of-movies-appearing-in-a-specific-sequence-of-genre

```py
class Solution:
    def __init__(self):
        self.map = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl", "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"}
    def letterCombinations(self, digits: str) -> List[str]:
        if len(digits) == 0:
            return []
        
        if len(digits) == 1:
            return self.map[digits[0]]
        
        result = deque(self.map[digits[0]])
        
        for digit in digits[1:]:
            size = len(result)
            
            for i in range(size):
                ele = result.popleft()
                
                for char in self.map[digit]:
                    result.append(ele + char)
        
        return result
```

## 6. Generate Parantheses
- https://leetcode.com/problems/generate-parentheses/description

**- Backtracking:**
A better approach is to use backtracking to generate only valid strings. This involves recursively building strings of length 2n and checking their validity as we go. 

To ensure that the current string is always valid during the backtracking process, we need two variables left_count and right_count that record the number of left and right parentheses in it, respectively.

Therefore, we can define our backtracking function as backtracking(cur_string, left_count, right_count) that takes the current string, the number of left parentheses, and the number of right parentheses as arguments. This function will build valid combinations of parentheses of length 2n recursively.

The function adds more parentheses to cur_string only when certain conditions are met:

    - If left_count < n, it suggests that a left parenthesis can still be added, so we add one left parenthesis to cur_string, creating a new string new_string = cur_string + (, and then call backtracking(new_string, left_count + 1, right_count).

    - If left_count > right_count, it suggests that a right parenthesis can be added to match a previous unmatched left parenthesis, so we add one right parenthesis to cur_string, creating a new string new_string = cur_string + ), and then call backtracking(new_string, left_count, right_count + 1).

```py
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:

        self.result = []

        def backtracking(leftcount, rightcount, currentstring):
            if len(currentstring) == 2 * n:
                self.result.append("".join(currentstring))
                return

            if leftcount < n:
                currentstring.append("(")
                backtracking(leftcount + 1, rightcount, currentstring)
                currentstring.pop()

            if rightcount < leftcount:
                currentstring.append(")")
                backtracking(leftcount, rightcount + 1, currentstring)
                currentstring.pop()


        backtracking(0, 0, [])

        return self.result
```

**Time complexity O(4^n/sqrt(n))**
Catalan Number - ask GPT for details!

**Space complexit O(n)**
The space complexity of a recursive call depends on the maximum depth of the recursive call stack, which is 2n. As each recursive call either adds a left parenthesis or a right parenthesis, and the total number of parentheses is 2n. Therefore, at most O(n)O(n)O(n) levels of recursion will be created, and each level consumes a constant amount of space.

## 7. Swap Nodes in Pairs
- https://leetcode.com/problems/swap-nodes-in-pairs/description

```py

# Recursive
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head == None or head.next == None:
            return head
        
        nextNode = head.next.next
        
        head, head.next = head.next, head
        
        head.next.next = self.swapPairs(nextNode)
        return head

# Iterative

class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        """
        :type head: ListNode
        :rtype: ListNode
        """
        # Dummy node acts as the prevNode for the head node
        # of the list and hence stores pointer to the head node.
        dummy = ListNode(-1)
        prev_node = dummy

        while head and head.next:

            # Nodes to be swapped
            first_node = head;
            second_node = head.next;

            # Swapping
            first_node.next = second_node.next
            second_node.next = first_node
            prev_node.next = second_node

            # Reinitializing the head and prev_node for next swap
            prev_node = first_node
            head = first_node.next

        # Return the new head node.
        return dummy.next
```
