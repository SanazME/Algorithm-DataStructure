"""
Given a binary tree, return the inorder traversal of its nodes' values.

Example:

Input: [1,null,2,3]
   1
    \
     2
    /
   3

Output: [1,3,2]
Follow up: Recursive solution is trivial, could you do it iteratively?

"""
# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        stack, result = [],[]
        while True:
            while root:
                stack.append(root)
                root = root.left
            if not stack:
                return result

            node = stack.pop()
            result.append(node.val)
            root = node.right

tree = TreeNode(1)
tree.right=TreeNode(2)
tree.right.left=TreeNode(3)
s =Solution()
print(s.inorderTraversal(tree))