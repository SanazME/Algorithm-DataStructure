"""
Print all paths from root to leaves in a binary tree
"""

def allPaths(root):
    if root is None:
        return 
    else:
        return helper(root, paths=[])
    
def helper(root, paths):
    if root is None:
        return
    else:
        paths.append(root.value)
        
        if isLeaf(root):
            print(paths)
            return
        else:
            helper(root.left, paths+[])
            helper(root.right, paths+[])

def isLeaf(node):
    return (not node.left) and (not node.right)

class BinaryTree(object):
    def __init__(self,value):
        self.value=value
        self.left = None
        self.right = None

tree = BinaryTree(10)
tree.left = BinaryTree(0)
tree.right = BinaryTree(25)

tree.left.left = BinaryTree(-1)
tree.left.right = BinaryTree(5)
tree.left.right.left=BinaryTree(4)
tree.left.right.right = BinaryTree(8)

tree.right.left = BinaryTree(16)
tree.right.right = BinaryTree(32)

tree.right.right.left = BinaryTree(28)
tree.right.right.right = None

allPaths(tree)