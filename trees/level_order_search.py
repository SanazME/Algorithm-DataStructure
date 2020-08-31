"""
Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level).

For example:
Given binary tree [3,9,20,null,null,15,7],
    3
   / \
  9  20
    /  \
   15   7
return its level order traversal as:
[
  [3],
  [9,20],
  [15,7]
]
"""

class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    def levelOrder(self, root):
        from collections import deque
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        
        queue = deque([root])
        result=[]

        while queue:
            size=len(queue)
            current_level =[]
            for i in range(size):
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
                current_level.append(node.val)
            result.append(current_level)
        return result


"""
Recurtion does not work for printing out in this format:
[[1],[3,4],[5,6]]
class Solution(object):
    def levelOrder(self, root):
        #from collections import deque
        
        if root is None:
            return []
        else:
            queue = []
            queue.insert(0,root)
            return [[root.val]]+self.levelOrderRecursion(queue)

    def levelOrderRecursion(self,queue,results=[]):
        if len(queue) == 0:
            return results
        root = queue.pop()
        subtree=[]
        if root.left:
            queue.insert(0,root.left)
            subtree.append(root.left.val)

        if root.right:
            queue.insert(0,root.right)
            subtree.append(root.right.val)

        if subtree:
            results.append(subtree)

        return self.levelOrderRecursion(queue)
"""      
s=Solution()

print(s.levelOrder(TreeNode(1)))

#[5,4,7,3,null,2,null,-1,null,9]
mm = TreeNode(5)
mm.left=TreeNode(4)
mm.left.left= TreeNode(3)
mm.left.left.left = TreeNode(-1)
mm.right = TreeNode(7)
mm.right.left = TreeNode(2)
mm.right.left.left = TreeNode(9)
print(s.levelOrder(mm))