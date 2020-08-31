"""
Given a binary tree, find its maximum depth.

The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

Note: A leaf is a node with no children.

Example:

Given binary tree [3,9,20,null,null,15,7],

    3
   / \
  9  20
    /  \
   15   7
return its depth = 3.


"""

# Definition for a binary tree node.
class TreeNode(object):
     def __init__(self, x):
         self.val = x
         self.left = None
         self.right = None

class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        if root is None:
            return 0
        else:

            leftBranch = self.maxDepth(root.left) + 1
            rightBranch = self.maxDepth(root.right)+ 1

            return max(leftBranch, rightBranch)

tree = TreeNode(3)
tree.left = TreeNode(9)
tree.left.left = None
tree.left.right = None
tree.right = TreeNode(20)
tree.right.left = TreeNode(15)
tree.right.right = TreeNode(7)

def print_tree(tree):
    if tree:
        print(tree.val)
        print_tree(tree.left)
        print_tree(tree.right)

print_tree(tree)

s = Solution()
s.maxDepth(tree)
