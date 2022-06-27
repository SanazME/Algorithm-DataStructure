# Feature #1: Design a system that can store and fetch words efficiently. This can be used to store web pages to make searching easier.
- https://leetcode.com/problems/implement-trie-prefix-tree/
- Trie: https://github.com/SanazME/Algorithm-DataStructure/blob/master/trees/README.md#trie
- Time complexity: `O(L)` for insertion and search
- Space complexity: `O(L)` for insertion but `O(1)` for search

# Feature #2: Design Search Autocomplete System
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
