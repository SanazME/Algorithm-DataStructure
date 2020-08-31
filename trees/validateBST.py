"""
Given a binary tree, determine if it is a valid binary search tree (BST).

Assume a BST is defined as follows:

The left subtree of a node contains only nodes with keys less than the node's key.
The right subtree of a node contains only nodes with keys greater than the node's key.
Both the left and right subtrees must also be binary search trees.

video: https://www.youtube.com/watch?v=MILxfAbIhrE

"""

class TreeNode(object):
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def isValidBST(root):
    """
    :type root: TreeNode
    :rtype: bool
    """
    def isValidBST_helper(root, min_val, max_val):
        if root is None:
            return True

        if (root.val <= min_val) or (root.val >= max_val):
            return False
        else:
            return isValidBST_helper(root.left, min_val, root.val)\
            and isValidBST_helper(root.right, root.val,max_val)

    return isValidBST_helper(root, -float('Inf'), float('Inf'))
