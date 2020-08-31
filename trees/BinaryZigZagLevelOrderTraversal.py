"""
Given a binary tree, return the zigzag level order traversal of its nodes' values. (ie, from left to right, then right to left for the next level and alternate between).

For example:
Given binary tree [3,9,20,null,null,15,7],
    3
   / \
  9  20
    /  \
   15   7
return its zigzag level order traversal as:
[
  [3],
  [20,9],
  [15,7]
]
"""
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        
class Solution(object):
    def zigzagLevelOrder(self, root):
        from collections import deque
        # Empty tree
        if root is None:
            return []
        queue = deque([root])
        result=[]
        k=1
        flag=1
        while queue:
            level=[]
            n = len(queue)
            
            for i in range(n):
                node = queue.pop()
                if node.left:
                    queue.appendleft(node.left)
                if node.right:
                    queue.appendleft(node.right)             
                level.append(node.val)
           
            result.append(level[::flag])
            flag *= -1
            
        return result  

s2=Solution()
kk = TreeNode(1)
kk.left = TreeNode(2)
kk.left.left = TreeNode(4)
kk.left.right = None
kk.right = TreeNode(3)
kk.right.right = TreeNode(5)
kk.right.left = None
print(s2.zigzagLevelOrder(kk))