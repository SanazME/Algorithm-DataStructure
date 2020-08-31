"""
Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).

For example, this binary tree [1,2,2,3,4,4,3] is symmetric:

    1
   / \
  2   2
 / \ / \
3  4 4  3
But the following [1,2,2,null,3,null,3] is not:
    1
   / \
  2   2
   \   \
   3    3
"""
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if root:
            return self.checkMirror(root.left, root.right)
        else:
            return True

    def checkMirror(self,subtreeL, subtreeR):
        # Both suntrees are empty
        if (subtreeL is None) and (subtreeR is None):
            return True
        # Both subtrees are not empty
        if subtreeL and subtreeR:
            return (subtreeL.val == subtreeR.val) and self.checkMirror(subtreeL.left, subtreeR.right) and self.checkMirror(subtreeL.right, subtreeR.left)

        return False
