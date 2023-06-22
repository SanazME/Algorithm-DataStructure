https://leetcode.com/list/?selectedList=o160a5j5#

## 1. Min Stack
### Solution 1
- Create an array and each element in the array is a set of `(val, minVal)` and this way we can track the min value in stack for every node.
```py
class MinStack:

    def __init__(self):
        self.stack = []
        

    def push(self, x: int) -> None:
        
        # If the stack is empty, then the min value
        # must just be the first value we add
        if not self.stack:
            self.stack.append((x, x))
            return

        current_min = self.stack[-1][1]
        self.stack.append((x, min(x, current_min)))
        
        
    def pop(self) -> None:
        self.stack.pop()
        

    def top(self) -> int:
        return self.stack[-1][0]
        

    def getMin(self) -> int:
        return self.stack[-1][1]
```
- **Time complexity: O(1) for all operations**
- **Space complexity: O(n)**


### Solution 2
- use a linked list and each node has a min val, we push and pop the head.
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

### Solution 3
- TWo stacks: Approach 1 required storing two values in each slot of the underlying Stack. Sometimes though, the minimum values are very repetitive. Do we actually need to store the same minimum value over and over again?
- Instead of only pushing numbers to the min-tracker Stack if they are less than the current minimum, we should push them if they are less than or equal to it. While this means that some duplicates are added to the min-tracker Stack, the bug will no longer occur. Here is another animation with the same test case as above, but the bug fixed.
```py
class MinStack:

    def __init__(self):
        self.stack = []
        self.min_stack = []        
        

    def push(self, x: int) -> None:
        self.stack.append(x)
        if not self.min_stack or x <= self.min_stack[-1]:
            self.min_stack.append(x)
    
    def pop(self) -> None:
        if self.min_stack[-1] == self.stack[-1]:
            self.min_stack.pop()
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]
```

### Solution 3 Improvement
- In the above approach, we pushed a new number onto the min-tracker Stack if, and only if, it was less than or equal to the current minimum.

One downside of this solution is that if the same number is pushed repeatedly onto MinStack, and that number also happens to be the current minimum, there'll be a lot of needless repetition on the min-tracker Stack. Recall that we put this repetition in to prevent a bug from occurring (refer to Approach 2).
An improvement is to put pairs onto the min-tracker Stack. The first value of the pair would be the same as before, and the second value would be how many times that minimum was repeated.

```py
class MinStack:

    def __init__(self):
        self.stack = []
        self.min_stack = []        
        

    def push(self, x: int) -> None:
        
        # We always put the number onto the main stack.
        self.stack.append(x)
        
        # If the min stack is empty, or this number is smaller than
        # the top of the min stack, put it on with a count of 1.
        if not self.min_stack or x < self.min_stack[-1][0]:
            self.min_stack.append([x, 1])
            
        # Else if this number is equal to what's currently at the top
        # of the min stack, then increment the count at the top by 1.
        elif x == self.min_stack[-1][0]:
            self.min_stack[-1][1] += 1

    
    def pop(self) -> None:

        # If the top of min stack is the same as the top of stack
        # then we need to decrement the count at the top by 1.
        if self.min_stack[-1][0] == self.stack[-1]:
            self.min_stack[-1][1] -= 1
            
        # If the count at the top of min stack is now 0, then remove
        # that value as we're done with it.
        if self.min_stack[-1][1] == 0:
            self.min_stack.pop()
            
        # And like before, pop the top of the main stack.
        self.stack.pop()


    def top(self) -> int:
        return self.stack[-1]


    def getMin(self) -> int:
        return self.min_stack[-1][0]   
```

## 2. Intersection of Two Linked Lists
- Time complexity: `O(N + M)` where `N` and `M` are lenght of linked lists
- Space complexity: `O(1)`

```py
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        pa, pb = headA, headB
        lena, lenb = 0, 0

        while pa.next:
            pa = pa.next
            lena += 1
        lena += 1

        while pb.next:
            pb = pb.next
            lenb += 1
        lenb += 1

        if pa.val != pb.val:
            return None

        if lena > lenb:
            pLong, pShort = headA, headB
            countLong, countShort = lena, lenb
        else:
            pLong, pShort = headB, headA
            countLong, countShort = lenb, lena
        
        while countLong > countShort:
            pLong = pLong.next
            countLong -= 1

        # both lists have the same length
        while pLong:
            if pLong == pShort:
                return pLong
            
            pLong = pLong.next
            pShort = pShort.next

        return None
```

## 3. Rotate Array
- https://github.com/SanazME/Algorithm-DataStructure/blob/master/array_stack_queues/README.md#rotate-array

## 4. Happy Number
- https://github.com/SanazME/Algorithm-DataStructure/blob/master/AmazonQuestions3.md#53-happy-number

## 5. Sum of two numbers
- https://leetcode.com/problems/sum-of-two-integers/editorial/?envType=list&envId=o160a5j5
```py
class Solution:
    def getSum(self, a: int, b: int) -> int:
        x, y = abs(a), abs(b)
        # ensure x >= y
        if x < y:
            return self.getSum(b, a)  
        sign = 1 if a > 0 else -1
        
        if a * b >= 0:
            # sum of two positive integers
            while y:
                x, y = x ^ y, (x & y) << 1
        else:
            # difference of two positive integers
            while y:
                x, y = x ^ y, ((~x) & y) << 1
        
        return x * sign
```

## 6. Find All Numbers Disappeared in an Array
- https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/description/?envType=list&envId=o160a5j5
- in O(1) space:
```py
def findMissingNumbers(nums):
    for i in range(n):
            curr = abs(nums[i])
            if nums[curr - 1] < 0:
                continue
            else:
                nums[curr - 1] *= -1

        missing = []
        for i in range(n):
            if nums[i] > 0:
                missing.append(i+1)

        return missing
```
  
## 7. Reverse Words in a String III
- https://leetcode.com/problems/reverse-words-in-a-string-iii/description/?envType=list&envId=o160a5j5
```py
class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        n = len(s)
        if n <= 1:
            return s

        count = 0
        out = ''

        while count < n:
            innerCount = count
            word = []
            while innerCount < n and s[innerCount] != ' ':
                word.append(s[innerCount])
                innerCount += 1

            # Reverse word
            out += self.reverseWord(word) + " "

            count = innerCount + 1

        out = out.strip()
        return out

    def reverseWord(self, word):
        out = ""
        for i in range(len(word)-1, -1, -1):
            out += word[i]
        
        return out
```

## 8. ** Subtree of another tree
- https://leetcode.com/problems/subtree-of-another-tree/editorial/?envType=list&envId=o160a5j5
**Inefficient first solution:**
Let's consider the most naive approach first. We can traverse the tree rooted at root (using Depth First Search) and for each node in the tree, check if the "tree rooted at that node" is identical to the "tree rooted at subRoot". If we find such a node, we can return true. If traversing the entire tree rooted at root doesn't yield any such node, we can return false.
Since we have to check for identicality, again and again, we can write a function isIdentical which takes two roots of two trees and returns true if the trees are identical and false otherwise.

Checking the identicality of two trees is a classical task. We can use the same approach as the one in Same Tree Problem. We can traverse both trees simultaneously and

- if any of the two nodes being checked is null, then for trees to be identical, both the nodes should be null. Otherwise, the trees are not identical.

- if both nodes are non-empty. Then for the tree to be identical, ensure that

    - values of the nodes are the same
    - left subtrees are identical
    - right subtrees are identical

**Algorithm**
1. Create a function dfs that takes node as the argument. This function will return true if the "tree rooted at node" is identical to the "tree rooted at subRoot" and false otherwise.

Now, if for any node, "tree rooted at node" is identical to the "tree rooted at subRoot", then we can be sure that there is a subtree in the "tree rooted at root" which is identical to the "tree rooted at subRoot".

2. For dfs,

- if node is null, we can return false because the null node cannot be identical to a tree rooted at subRoot, which as per constraints is not null.

- else if, check for the identicality of the "tree rooted at node" and the "tree rooted at subRoot" using the function isIdentical, if trees are identical, return true.

- otherwise, call the dfs function for the left child of node, and the right child of node. If either of them returns true, return true. Otherwise, return false.

3. Create the function isIdentical which takes two roots of two trees (namely node1 and node2) and returns true if the trees are identical and false otherwise.

```py
class Solution:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:

        # Check for all subtree rooted at all nodes of tree "root"
        def dfs(node):

            # If this node is Empty, then no tree is rooted at this Node
            # Thus "tree rooted at node" cannot be same as "rooted at subRoot"
            # "tree rooted at subRoot" will always be non-empty (constraints)
            if node is None:
                return False

            # If "tree rooted at node" is identical to "tree at subRoot"
            elif is_identical(node, subRoot):
                return True

            # If "tree rooted at node" was not identical.
            # Check for tree rooted at children
            return dfs(node.left) or dfs(node.right)

        def is_identical(node1, node2):

            # If any one Node is Empty, both should be Empty
            if node1 is None or node2 is None:
                return node1 is None and node2 is None

            # Both Nodes Value should be Equal
            # And their respective left and right subtree should be identical
            return (node1.val == node2.val and
                    is_identical(node1.left, node2.left) and
                    is_identical(node1.right, node2.right))

        # Check for node rooted at "root"
        return dfs(root)
```

**Time and Space complexities**
- Time: `O(NM)`. For every `N` node in the tree, we check if the tree rooted at node is identical to subRoot. This check takes `O(M)` time, where `M` is the number of nodes in `subRoot`.
- Space: `O(N + M)`. There will be at most `N` recursive call to `dfs` (or `isSubtree`). Now, each of these calls will have `M` recursive calls to `isIdentical`. Before `isIdentical`, our call stack has at most `O(N)` elements and might increase to `O(N+M)` during the call. After calling `isIdentical`, it will be back to at most `O(N)` since all elements made by `isIdentical` are popped out.

**Efficient solution with Hasing nodes**
- It turns out that tree comparison is expensive. In the very first approach, we need to perform the comparison for at most NNN nodes, and each comparison cost `O(M)`. If we can somehow reduce the cost of comparison, then we can reduce the overall time complexity

You may recall that the cost of comparison of two integers is constant. As a result, if we can somehow transform the subtree rooted at each node to a unique integer, then we can compare two trees in constant time.

*Is there any way to transform a tree into an integer?
Yes, there is. We can use the concept of Hashing.*

- We want to hash (map) each subtree to a unique value. We want to do this in such a way that if two trees are identical, then their hash values are equal. And, if two trees are not identical, then their hash values are not equal. This hashing can be used to compare two trees in O(1) time.

We will build the hash of each node depending on the hash of its left and right child. The hash of the root node will represent the hash of the whole tree because to build the hash of the root node, we used (directly, or indirectly) the hash values of all the nodes in its subtree.

If any node in "tree rooted at root" has hash value equal to the hash value of "tree rooted at subRoot", then "tree rooted at subRoot" is a subtree of "tree rooted at root", provided our hashing mechanism maps nodes to unique values.

- One can use any hashing function which guarantees minimum spurious hits and is calculated in O(1) time. We will use the following hashing function.
    - if it's `null` node, then hash it to `3`. (you can use any prime number here)
    - Else,
        - left shift the hash value of the left node by some fixed value
        - left shift the hash value of right node by 1
        - add these shifted values with this `node.val` to get the hash of this node
     
- avoid concatenating strings for hash value purposes because it will take `O(N)` time to concatenate strings.
- To ensure minimum spurious hits, we can map each node to two hash values, thus getting one hash pair for each node. Trees rooted at s and Tree rooted at t will have the same hash pair iff they are identical, provided our hashing technique maps nodes to unique hash pairs

**Algorithm**
1. Define two constants MOD_1 and MOD_2 as two large prime numbers.

2. Initialize a memo set to memoize hash values of each node in the tree rooted at root. Please ensure that the data structure for memoizing has constant time insertion.

3. Define function hashSubtreeAtNode which takes two parameter, node and needToAdd. It essentially returns the hash pair of the subtree rooted at node.

- If node is null, then return hash pair (3, 7). Note that we can return any two values.

- Else, compute the hash pair of this node using the left and right child's hash pair.

- For the first hash of the pair

    - left shift the first hash of the left node by some fixed value.
    - left shift the first hash of the right node by 1 (or some other fixed value)
    - add these shifted values with this node's value to get the first hash of this node.
    - take MOD_1 at each step to avoid overflow.

- For the second hash of the pair

    - left shift the second hash of the left node by some fixed value (different from what was used for the first element of the pair).
    - left shift the second hash of the right node by 1 (or some other fixed value)
    - add these shifted values with this node's value to get the second hash of this node.
    - take MOD_2 at each step to avoid overflow.

- If needToAdd is true, then add this hash pair to memo.

- Return the hash pair of this node.

4. Call hashSubtreeAtNode(root, true), for calculating the hash of root, we will calculate the hash of every node in the tree rooted at root. The true value of needToAdd means we will add every computed hash to the memo.

5. Now, call hashSubtreeAtNode(subRoot, false). This will calculate the hash of subRoot and will not add it to the memo. If the hash pair of subRoot is present in memo, then subRoot is a subtree of root. Hence, return if hashSubtreeAtNode(subRoot, false) is present in memo or not.

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

**Time and Space complexity**
- Time: `O(N+M)`
- We are traversing the tree rooted at root in O(N) time. We are also traversing the tree rooted at subRoot in O(M) time. For each node, we are doing constant time operations. After traversing, for lookup we are either doing O(1) operations, or O(N) operations. Hence, the overall time complexity is O(N+M).
- Space: `O(N+M)`
- We are using memo to store the hash pair of each node in the tree rooted at root. Hence, for this, we need O(N) space.
Moreover, since we are using recursion, the space required for the recursion stack will be O(N) for hashSubtreeAtNode(root, true) and O(M) for hashSubtreeAtNode(subRoot, false).
Hence, overall space complexity is O(M+N).

## 9. Can Place Flowers
- https://github.com/SanazME/Algorithm-DataStructure/blob/master/AmazonQuestions3.md#56-can-place-flowers
