# Definition for a binary tree node.
#class TreeNode:
#    def __init__(self, x):
#        self.val = x
#        self.left = None
#        self.right = None

"""
Algorithm
The inverse of an empty tree is the empty tree.
The inverse of a tree with root r, and subtrees right and left,
is a tree with root r, whose right subtree is the inverse of
left, and whose left subtree is the inverse of right.
"""
class Solution:
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if root:
            invert = self.invertTree
            root.left, root.right = invert(root.right), invert(root.left)
            return root

#And an iterative version using my own stack:
def invertTree2(self, root):
    stack = [root]
    while stack:
        node = stack.pop()
        if node:
            node.left, node.right = node.right, node.left
            stack += node.left, node.right
    return root
