class Solution(object):
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        result = []
        
        if root is None:
            return []
        
        stack=[]
        while True:
            
            while root:
                print(root.val)
                stack.append(root)
                root = root.left
            if not stack:
                return result
            
            node=stack.pop()
            root = node.right