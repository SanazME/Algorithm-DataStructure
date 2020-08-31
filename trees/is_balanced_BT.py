"""
Given a binary tree, determine if it is height-balanced.

For this problem, a height-balanced binary tree is defined as:

a binary tree in which the depth of the two subtrees of every node never differ by more than 1.

Example 1:

Given the following tree [3,9,20,null,null,15,7]:

    3
   / \
  9  20
    /  \
   15   7
Return true.

Example 2:

Given the following tree [1,2,2,3,3,null,null,4,4]:

       1
      / \
     2   2
    / \
   3   3
  / \
 4   4
Return false.

SOlution: https://www.afternerd.com/blog/python-check-tree-balanced/

a simple but slower solution involves two recursions:
From the definition of a balanced tree, we can conclude that a binary tree is balanced if:

1- the right subtree is balanced

2- the left subtree is balanced

3- the difference between the height of the left subtree and the right subtree is at most 1

def is_balanced(root):
    if not root:
        return True
    return is_balanced(root.left) and is_balanced(root.right) and \
        abs(helper(root.left)-helper(root.right)) <= 1

def helper(root):
    if not root:
        return 0
    else:
        return max(helper(root.left), helper(root.right))+1




In the following solution:
Let’s redefine our recursive function is_balanced_helper to be a function that takes one argument,
 the tree root, and returns an integer such that:

1- if the tree is balanced, return the height of the tree

2- if the tree is not balanced, return -1

Notice that this new is_balanced_helper can be easily implemented recursively as well by following
 these rules:

1- apply is_balanced_helper on both the right and left subtrees

2- if either the right or left subtrees returns -1, then we should return -1 (because our tree is 
obviously not balanced if either subtrees is not balanced)

3- if both subtrees return an integer value (indicating the heights of the subtrees), then we check
 the difference between these heights. If the difference doesn’t exceed 1, then we return the height of this tree. Otherwise, we return -1

"""

def is_balanced(root):
    return is_balanced_helper(root) > -1

def is_balanced_helper(root):
    if not root:
        return 0
    # if left subtree is not balanced return -1
    left_height = is_balanced_helper(root.left)
    if left_height == -1:
        return -1
    # if right subtree is not balanced return -1
    right_height = is_balanced_helper(root.right)
    if right_height == -1:
        return -1
    
    # If neither left tree and nor right tree are unbalanced, get the difference
    # between their heights, if it is > 1, return -1 else return the height of the main tree
    if abs(left_height - right_height) > 1:
        return -1
    else:
        return max(right_height, left_height)+1


class TreeNode(object):
    def __init__(self,value):
        self.value = value
        self.right=None
        self.left = None   
tree=TreeNode(3)
tree.left=TreeNode(5)
tree.left.left = TreeNode(2)
tree.left.right = TreeNode(6)
tree.left.left.left = TreeNode(0)
tree.left.left.right = None
tree.right = TreeNode(78)

print(is_balanced(tree))
