## Tree

- Linear data structures like as arrays, linked lists, queues and stack, data is ordered and you traverse them sequentially (pic: https://medium.com/basecs/how-to-not-be-stumped-by-trees-5f36208f68a7)
- Non-linear data structure can be trees and the data traversal is not sequential. Trees are made of nodes, links. A root, parent nodes, sibling nodes (the same parent), inner nodes, leaf nodes.
- **Depth** of a node: how many edges from the node to the root 
- **Height** of a node: the maximum path from the node to the furthest leaf node
- so the height of the root is the height of the tree.
- trees are recursive data structures, a tree has a nested subtrees.
- **balanced trees**: if any two **sibling subtree** do not differ in the height by more than one level.(otherwise it will be an unbalanced tree)
- filesystems are trees

**Binary Search Tree (BST)**
1. BST, every parent node can only have two possible child nodes and not more than that
2. so the root node points to two subtrees, left subtree and right subtree and this recursively is true for each subtrees that are binary trees and two subtrees...
3. the searchable part of the BST is that all the subtrees to the left a node are smaller than the value of that node and all the subtrees to the right of that node are larger in value than the value of that node.
- like the indexing databases are BST. or **git bisect** to find where the bad commit happens uses BST.
- BST are fast in insertion and lookup - **average case** a BST algorithm can insert or locate a node in a n-node BST in log2(N). The worst case can be slower based on the shapre of the tree (skewed)
- Always checkout the base/edge cases: empty tree (Null) and one node, two nodes and very skewed trees (https://yangshun.github.io/tech-interview-handbook/algorithms/tree)
- 



### Height-balanced Binary Tree
[**Drill Down With Recursion And Respond Back Up**](https://www.youtube.com/watch?v=LU4fGD-fgJQ) : 

- We can notice that we don't need to know the heights of all of the subtrees all at once. All we need to know is whether a subtree is height balanced or not and the height of the tree rooted at that node, not information about any of its descendants. **Our base case is that a null node (we went past the leaves in our recursion) is height balanced and has a height of -1 since it is an empty tree (under the sea)**. So the key is that we will drive towards our base case of the null leaf descendant and deduce and check heights on the way upwards.

- **Key points of interest:**
  1. Is the subtree height balanced?
  2. What is the height of the tree rooted at that node?

**Complexities**

**Time: O( n )**
- This is a postorder traversal (left right node) with possible early termination if any left subtree turns out unbalanced and an early result bubbles back up.
At worst we will still touch all n nodes if we have no early termination.

**Space: O( h )**
- Our call stack (from recursion) will only go as far deep as the height of the tree, so h (the height of the tree) is our space bound for the amount of call stack frames that we will create.

1. Example 1: check if the tree is height-balanced:
```py
class TreeNode(object):
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

root = [1,2,2,3,3,None,None,4,4]
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)
root.left.left.left = TreeNode(6)
root.left.left.right = TreeNode(7)

def isBalanced(node):
    if node is None:
        return (-1, True)
    
    leftHeight, isLeftSubtreeBalanced = isBalanced(node.left)
    rightHeight, isRightSubtreeBalanced = isBalanced(node.right)
    height = max(leftHeight, rightHeight) + 1
    isSubtreeBalanced = isLeftSubtreeBalanced and isRightSubtreeBalanced and abs(leftHeight - rightHeight) <= 1
    print('For Node:{} height:{} and balanced: {}'.format(node.val, height, isSubtreeBalanced))
    return (height, isSubtreeBalanced)
    
print(isBalanced(root))
```
### Traverse a Tree 

**1. Pre-order Tree Traversal**: Pre-order traversal is to visit the root first. Then traverse the left subtree. Finally, traverse the right subtree.
**In iterative method, we want to make sure that the left node is popped before the right node**:

```py
class TreeNode(object):
  def __init__(self, val):
      self.val = val
      self.left = None
      self.right = None

# Recursive
def preorder(root):
  return helper(root, result=[])
  
def helper(root, result):
  if root == None: 
     return result
     
  result.append(root.val)
  helper(root.left, result)
  helper(root.right, result)
  return result
  
# Iterative
def preorder(root):
  result = []
  if root == None: return result
  
  stack = [root]
  
  while stack:
    node = stack.pop()
    result.append(node.val)
    
    if node.right: stack.append(node.right)
    if node.left: stack.append(node.left)
    
  return result
```

**2. In-order Traversal**: In-order traversal is to traverse the left subtree first. Then visit the root. Finally, traverse the right subtree. Typically, **for binary search tree, we can retrieve all the data in sorted order using in-order traversal.**
```py
# recursive
def inorder(root):
  return helper(root, result=[])
  
def helper(root, result):
  if root == None: 
    return result
    
  helper(root.left, result)
  result.append(root.val)
  helper(root.right, result)
  
  return result
  
# iterative
def inorder(root):
  result = []
  if root == None: return result
  
  stack = []
  
  while True:
    while root:
      stack.append(root)
      root = root.left
      
    if len(stack) == 0:
      return result
      
    node = stack.pop()
    result.append(node.val)
    root = node.right
```

**3. Post-order Traversal**: Post-order traversal is to traverse the left subtree first. Then traverse the right subtree. Finally, visit the root. It is worth noting that **when you delete nodes in a tree, deletion process will be in post-order. That is to say, when you delete a node, you will delete its left child and its right child before you delete the node itself.**

Also, **post-order is widely use in mathematical expression. It is easier to write a program to parse a post-order expression. you can easily handle the expression using a stack. Each time when you meet a operator, you can just pop 2 elements from the stack, calculate the result and push the result back into the stack.**

```py
# Iterative
def postorder(root):
  result = []
  if root == None: return result
  
  stack = []
  visited = set()
  
  while True:
    while root:
      stack.append(root)
      root = root.left

    if not stack: return result
    node = stack[-1]

    if node.right and node not in visited:
      visited.add(node)
      root = node.right
    else:
      node2 = stack.pop()
      result.append(node2.val)
    
```

### Solve Tree problems recursively
- **Top-dowm solution**: "Top-down" means that in each recursive call, we will visit the node first to come up with some values, and pass these values to its children when calling the function recursively. So **the "top-down" solution can be considered as a kind of preorder traversal. To be specific, the recursive function top_down(root, params)**.
- **"Bottom-up"** is another recursive solution. In each recursive call, we will firstly call the function recursively for all the children nodes and then come up with the answer according to the returned values and the value of the current node itself. This process can be regarded as a kind of postorder traversal. 
- When you meet a tree problem, ask yourself two questions: Can you determine some parameters to help the node know its answer? Can you use these parameters and the value of the node itself to determine what should be the parameters passed to its children? If the answers are both yes, try to solve this problem using a **"top-down" recursive solution.**
- Or, you can think of the problem in this way: for a node in a tree, if you know the answer of its children, can you calculate the answer of that node? If the answer is yes, solving the problem recursively using a **bottom up** approach might be a good idea.
- **Bottom-up approach to find the max depth of a binary tree**:
```py
class TreeNode(object):
  def __init__(self, val):
      self.val = val
      self.left = left
      self.right = right

def maxDepth(root):
  if root == None:
      return 0
      
  left-depth = maxDepth(root.left)
  right-depth = maxDepth(root.right)
  depth = max(left-depth, right-depth) + 1
  
  return depth
```

- **Top-down approach:**
```py
def maxDepth(root):
  if root == None:
    return 0
    
  self.answer = 0
  def probe(node, depth):
    # leaf node
    if node.left == None and node.right == None
        self.answer = max(self.answer, depth + 1)
    if node.left: probe(node.left, depth + 1)
    if node.right: probe(node.right, depth + 1)  
  
  probe(root, 0)
  return self.answer
```

### Symmetric Tree
- Two trees are a mirror reflection of each other if:

1. Their two roots have the same value.
2. The right subtree of each tree is a mirror reflection of the left subtree of the other tree.

**The Recursive Complexity Analysis:**

- *Time complexity* : O(n). Because we traverse the entire input tree once, the total run time is O(n), where nn is the total number of nodes in the tree.

- *Space complexity* : **The number of recursive calls is bound by the height of the tree**. In the worst case, the tree is linear and the height is in O(n)O(n). Therefore, space complexity due to recursive calls on the stack is O(n)O(n) in the worst case.

** The Iterative Complexity**
- *Time complexity* : O(n). Because we traverse the entire input tree once, the total run time is O(n)O(n), where nn is the total number of nodes in the tree.

- *Space complexity* : There is additional space required for the search queue. In the worst case, we have to insert O(n) nodes in the queue. Therefore, space complexity is O(n).

- For iterative call, we use BFS & queue and compare two adjacent nodes to see if the symmtric condition holds. At the beginning, we need to push root node twice to pass the comparison in the loop.

### Populating Next Right Pointers in Each Node (Leetcode)
- Since we are manipulating tree nodes on the same level, it's easy to come up with
a very standard BFS solution using queue. But because of next pointer, we actually
don't need a queue to store the order of tree nodes at each level, we just use a next
pointer like it's a link list at each level; In addition, we can borrow the idea used in
the Binary Tree level order traversal problem, which use cur and next pointer to store
first node at each level; we exchange cur and next every time when cur is the last node
at each level.

- https://leetcode.com/explore/learn/card/data-structure-tree/133/conclusion/1016/:
- when we're given a tree with tree nodes having `.next`, It's a BFS traversal. now pointer is the current level traveler and head is the left most element at next level and the tail is the right most element at next level till now. We move now pointer at current level and populate the the next-link at its children level.

```py
def connect(self, root):
        """
        :type root: Node
        :rtype: Node
        """
        curr = root
        head = tail = None
        
        while curr:
            if curr.left:
                if tail:
                    tail.next = curr.left
                    tail = tail.next
                else:
                    head = tail = curr.left
            if curr.right:
                if tail:
                    tail.next = curr.right
                    tail = tail.next
                else:
                    head = tail = curr.right

            if not curr.next:
                curr = head
                head = tail = None
            else:
                curr = curr.next
        return root
   ```
