## Feature #1: Design a system that can store and fetch words efficiently. This can be used to store web pages to make searching easier.
- https://leetcode.com/problems/implement-trie-prefix-tree/
- Trie: https://github.com/SanazME/Algorithm-DataStructure/blob/master/trees/README.md#trie
- Time complexity: `O(L)` for insertion and search
- Space complexity: `O(L)` for insertion but `O(1)` for search

## Feature #2: Design Search Autocomplete System
- https://leetcode.com/problems/design-search-autocomplete-system/
- The following solution is slightly different than the solution further below. In here, we also have to sort based on ASCII-code and the difference bw this and the further down solution is that here:
  - we just add words to suggestions list. The words can be repeated (we don't use `set` here to get unique words) 
  - The suggestion list only includes added words and not the frequency and its lenght can get big and not limited to 3 elements
```py
class TrieNode:
    def __init__(self):
        self.children = {}
        self.endOfWord = False
        self.suggestions = []

class AutocompleteSystem(object):

    def __init__(self, sentences, times):
        """
        :type sentences: List[str]
        :type times: List[int]
        """
        self.root = TrieNode()
        self.keywords = ""
        self.cach_count = dict()
        
        for i, sentence in enumerate(sentences):
            self._addResource(sentence, times[i])
            self.cach_count[sentence] = times[i]
            
    
    def _addResource(self, word, hot):
        curr = self.root
        
        for char in word:
            if char not in curr.children:
                curr.children[char] = TrieNode()
        
            curr = curr.children[char]
            curr.suggestions.append(word)

        
        curr.endOfWord = True
      
    
    def search(self, word):
        curr = self.root
        
        for char in word:
            if char not in curr.children:
                return []
            curr = curr.children.get(char)
            
        return curr.suggestions
    

    def input(self, c):
        """
        :type c: str
        :rtype: List[str]
        """
        
        if c != "#":
            self.keywords += c
            words = self.search(self.keywords)
            res = []
            for word in words:
                res.append((self.cach_count[word], word))
            
            print(res)
            res = list(set(res))
            
            return [s[1] for s in sorted(res, key=lambda x: (-x[0], x[1]))[:3]]
        else:
            self._addResource(self.keywords, 1)
            self.cach_count[self.keywords] = self.cach_count.get(self.keywords, 0) + 1
            self.keywords = ""
            
        return []
        
        


# Your AutocompleteSystem object will be instantiated and called as such:
# obj = AutocompleteSystem(sentences, times)
# param_1 = obj.input(c)
```


- https://leetcode.com/problems/search-suggestions-system/
- 
- https://github.com/SanazME/Algorithm-DataStructure/blob/master/AmazonQuestions2.md#38-design-search-autocomplete-system

**Other Solution**
```py
class TrieNode():
    def __init__(self):
        self.isEnd = False
        self.children = {}
        self.hot = 0
    
class AutocompleteSystem(object):
    def __init__(self, sentences, times):
        self.root = TrieNode()
        self.searchTerm = ""
        # 1. add historical data
        for i, sentence in enumerate(sentences):
            self.add(sentence, times[i])
			
    def add(self,sentence, hot):
        node = self.root
        #2. for each character in sentence
        for c in sentence: 
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        #3. when last character is added,
        #   make node.isEnd = True indicate that the current node is end of the sentence
        node.isEnd = True
        #4. do -= because by negating, we can sort as ascending order later
        node.hot-= hot
        
    def search(self):
        node = self.root
        res = []
        path = ""

        for c in self.searchTerm:
            if c not in node.children:
                return res
            # 6. add each character to path variable, path will added to res when we found node.isEnd ==True
            path +=c
            node = node.children[c]
        # 7. at this point, node is at the given searchTerm.
        # for ex. if search term is "i_a", we are at "a" node.
        # from this point, we need to search all the possible sentence by using DFS
        self.dfs(node, path,res)
        # 11. variable res has result of all the relevant sentences
        # we just need to do sort and return [1] element of first 3
        return [item[1] for item in sorted(res)[:3]]
            
    def dfs(self,node, path, res):
        # 8. Check if node is end of the sentence
        # if so, add path to res
        if node.isEnd:
            # 9. when add to res, we also want to add hot for sorting
            res.append((node.hot,path))
        # 10. keep going if the node has child
        # until there is no more child (reached to bottom)
        for c in node.children:
            self.dfs(node.children[c], path+c,res)

    def input(self, c):
        if c != "#":
            # 5. if input is not "#" add c to self.searchTerm and do self.search each time
            self.searchTerm +=c
            return self.search()
        
        else:
            self.add(self.searchTerm, 1)
            self.searchTerm =""
```

- https://leetcode.com/problems/design-add-and-search-words-data-structure/
```py
class TrieNode:
    def __init__(self):
        self.children = {}
        self.endOfWord = False

class WordDictionary:

    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        curr = self.root
        
        for char in word:
            if char not in curr.children:
                curr.children[char] = TrieNode()

            curr = curr.children[char]
            
        curr.endOfWord = True
        

    def search(self, word: str) -> bool:
        curr = self.root
        return self._search(word, curr)
        
        
    def _search(self, word, curr):
        
        for i, char in enumerate(word):
            if char not in curr.children:
                if char == '.':
                    for x in curr.children:
                        if self._search(word[i + 1:], curr.children[x]):
                            return True
                return False
            else:
                curr = curr.children[char]
                
        return curr.endOfWord
```
- Time complexity:
  - Add a word: `O(M)` where M is the key length. At each step, we either examine or create a node in the trie. That makes only M operations
  - Search a word: `O(M)` for a well-defined words without dots. The worst case: `O(N * 26^M)` where N is the number of keys (number of nodes in trie) and M is the word length. We need to search `O(26^M)` at each node.
- Space complexity: 
  - Add a word: `O(M)`. In the worst case newly inserted key doesn't share a prefix with the keys already inserted in the trie. We have to add M new nodes.


## Feature #3: Add White Spaces to Create Words
- https://leetcode.com/problems/word-break/
- https://github.com/SanazME/Algorithm-DataStructure/blob/master/AmazonQuestions2.md#39-word-break
- The above solution does to work for simple case of `"a", ["a"]`. We need to DP and set the first index of dp to True because "" is a valid string.
- The intuition behind this approach is that the given problem (s) can be divided into subproblems s1 and s2. If these subproblems individually satisfy the required conditions, the complete problem, s also satisfies the same. e.g. "catsanddog" can be split into two substrings "catsand", "dog". The subproblem "catsand" can be further divided into "cats","and", which individually are a part of the dictionary making "catsand" satisfy the condition. Going further backwards, "catsand", "dog" also satisfy the required criteria individually leading to the complete string "catsanddog" also to satisfy the criteria.

Now, we'll move onto the process of `dp` array formation. We make use of `dp` array of size `n+1`, where `n` is the length of the given string. We also use two index pointers i and j, where i refers to the length of the substring `s′` considered currently starting from the beginning, and j refers to the index partitioning the current substring `s′` into smaller substrings `s′(0,j)` and `s′(j+1,i)`.

To fill in the `dp` array, we initialize the element `dp[0]` as true, since the null string is always present in the dictionary, and the rest of the elements of `dp` as false. We consider substrings of all possible lengths starting from the beginning by making use of index i. For every such substring, we partition the string into two further substrings `s1'` and `s2'` in all possible ways using the index j (Note that the i now refers to the ending index of `s2'`. Now, to fill in the entry `dp[i]`, we check if the `dp[j]` contains `true`, i.e. if the substring `s1'` fulfills the required criteria. If so, we further check if `s2'` is present in the dictionary. If both the strings fulfill the criteria, we make `dp[i]` as `true`, otherwise as `false`.

```py
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        wordSet = set(wordDict)
        cache = {}
        
        dp = [False for _ in range(len(s) + 1)]
        dp[0] = True
        
        for i in range(1, len(s) + 1):
            for j in range(i):
                if dp[j] and s[j:i] in wordSet:
                    dp[i] = True
                    break
        return dp[-1]
```

- Time complexity: `O(n^3)` There are two nested loops and substring computation at each iteration. 
- Space complexity: `O(n)`. Length of p array is n+1.

## Feature #4: Suggest Possible Queries After Adding White Spaces
- https://leetcode.com/problems/word-break-ii/
- The solutions for this problem go by many names, such as Dynamic Programming, recursion with memoization, DFS, and backtracking etc. They all capture certain traits of the solutions.

In essence, all these solutions can all be categorized as variants of Dynamic Programming (DP), as we will discuss in this article.

As a reminder, with DP, we break the original problem down to several sub-problems recursively until the sub-problems are small enough to be solved directly. Then we combine the results of sub-problems to obtain the final solution for the original problem.

As one can see, the DP solutions are also the embodiment of the divide-and-conquer principle.

To come up a DP solution, the essential step is to represent the solution of the original problem with the results of its sub-problems. In general, there are two approaches to implement a DP solution, namely Top-Down and Bottom-Up. We would explain in detail how to apply these two approaches to this problem in the following sections.The solutions for this problem go by many names, such as Dynamic Programming, recursion with memoization, DFS, and backtracking etc. They all capture certain traits of the solutions.

In essence, all these solutions can all be categorized as variants of Dynamic Programming (DP), as we will discuss in this article.

As a reminder, with DP, we break the original problem down to several sub-problems recursively until the sub-problems are small enough to be solved directly. Then we combine the results of sub-problems to obtain the final solution for the original problem.

As one can see, the DP solutions are also the embodiment of the divide-and-conquer principle.

To come up a DP solution, the essential step is to represent the solution of the original problem with the results of its sub-problems. In general, there are two approaches to implement a DP solution, namely Top-Down and Bottom-Up. We would explain in detail how to apply these two approaches to this problem in the following sections.

**Approach 1: Top-Down Dynamic Programming**

**Algorithm**

Following the above intuition, it seems intuitive to implement the solution with recursion.

We define a recursive function called _wordBreak_topdown(s) which generates the results for the input string. Here are a few steps to implement our recursive function.

- First of all, as the base case of the recursion, when the input string is empty, the recursion would terminate. Note that we return a list of empty list as the result, rather than just an empty list.

- As the main body of the function, we run an iteration over all the prefixes of the input string. If the corresponding prefix happens to match a word in the dictionary, we then invoke recursively the function on the postfix.

- At the end of the iteration, we keep the results in the hashmap named memo with each valid postfix string as its key and the list of words that compose the prefix of as the value. For instance, for the postfix dogo, its corresponding entry in the hashmap would be memo["dogo"] = ["do", "go"].

```py
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        wordSet = set(wordDict)
        # table to map a string to its corresponding words break
        # {string: [['word1', 'word2'...], ['word3', 'word4', ...]]}
        memo = defaultdict(list) # memo = {}

        #@lru_cache(maxsize=None)    # alternative memoization solution
        def _wordBreak_topdown(s):
            """ return list of word lists """
            if not s:
                return [[]]  # list of empty list

            if s in memo:
                # returned the cached solution directly.
                return memo[s]

	    # memo[s] = []
            for endIndex in range(1, len(s)+1):
                word = s[:endIndex]
                if word in wordSet:
                    # move forwards to break the postfix into words
                    for subsentence in _wordBreak_topdown(s[endIndex:]):
                        memo[s].append([word] + subsentence)
            return memo[s]

        # break the input string into lists of words list
        _wordBreak_topdown(s)

        # chain up the lists of words into sentences.
        return [" ".join(words) for words in memo[s]]
```
- Time complexity: `O(N^2 + 2^N + W)` refer to https://leetcode.com/problems/word-break-ii/solution/ for explaination
- Space complexity: `O(2^N * N + W)` (https://salonikaurone.medium.com/leetcode-word-break-ii-explained-d41ecfbe8fc5)

## Feature #5: Calculate the Search Ranking Factor
- https://leetcode.com/problems/product-of-array-except-self/
```py
# with O(N) space
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        
        # The length of the input array 
        length = len(nums)
        
        # The left and right arrays as described in the algorithm
        L, R, answer = [0]*length, [0]*length, [0]*length
        
        # L[i] contains the product of all the elements to the left
        # Note: for the element at index '0', there are no elements to the left,
        # so the L[0] would be 1
        L[0] = 1
        for i in range(1, length):
            
            # L[i - 1] already contains the product of elements to the left of 'i - 1'
            # Simply multiplying it with nums[i - 1] would give the product of all 
            # elements to the left of index 'i'
            L[i] = nums[i - 1] * L[i - 1]
        
        # R[i] contains the product of all the elements to the right
        # Note: for the element at index 'length - 1', there are no elements to the right,
        # so the R[length - 1] would be 1
        R[length - 1] = 1
        for i in reversed(range(length - 1)):
            
            # R[i + 1] already contains the product of elements to the right of 'i + 1'
            # Simply multiplying it with nums[i + 1] would give the product of all 
            # elements to the right of index 'i'
            R[i] = nums[i + 1] * R[i + 1]
        
        # Constructing the answer array
        for i in range(length):
            # For the first element, R[i] would be product except self
            # For the last element of the array, product except self would be L[i]
            # Else, multiple product of all elements to the left and to the right
            answer[i] = L[i] * R[i]
        
        return answer
```
**Algorithm**
1. Initialize the empty answer array where for a given index i, answer[i] would contain the product of all the numbers to the left of i.

2. We construct the answer array the same way we constructed the L array in the previous approach. These two algorithms are exactly the same except that we are trying to save up on space.

3. The only change in this approach is that we don't explicitly build the R array from before. Instead, we simply use a variable to keep track of the running product of elements to the right and we keep updating the answer array by doing answer[i] = answer[i] * R. For a given index i, answer[i] contains the product of all the elements to the left and R would contain product of all the elements to the right. We then update R as R = R * nums[i].

```py
# with O(1) space
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        
        # The length of the input array 
        length = len(nums)
        
        # The answer array to be returned
        answer = [0]*length
        
        # answer[i] contains the product of all the elements to the left
        # Note: for the element at index '0', there are no elements to the left,
        # so the answer[0] would be 1
        answer[0] = 1
        for i in range(1, length):
            
            # answer[i - 1] already contains the product of elements to the left of 'i - 1'
            # Simply multiplying it with nums[i - 1] would give the product of all 
            # elements to the left of index 'i'
            answer[i] = nums[i - 1] * answer[i - 1]
        
        # R contains the product of all the elements to the right
        # Note: for the element at index 'length - 1', there are no elements to the right,
        # so the R would be 1
        R = 1;
        for i in reversed(range(length)):
            
            # For the index 'i', R would contain the 
            # product of all elements to the right. We update R accordingly
            answer[i] = answer[i] * R
            R *= nums[i]
        
        return answer
```
- Time complexity: `Q(N)`
- Space complexity: `O(N) or O(1)`
