"""
How do you count a number of leaf nodes in a given binary tree?
"""

def countLeafNodes(root, count=0):
    if root is None:
        return 0
    else:
        if isLeaf(root):
            return 1
        else:
            return countLeafNodes(root.left, count) + countLeafNodes(root.right, count)

def isLeaf(node):
    return (not node.left) and (not node.right)

class BinaryTree(object):
    def __init__(self,value):
        self.value = value
        self.left = None
        self.right = None

tree = BinaryTree(10)
tree.left = BinaryTree(0)
tree.right = BinaryTree(25)

tree.left.left = BinaryTree(-1)
tree.left.right = None

tree.right.left = BinaryTree(16)
tree.right.right = BinaryTree(32)

tree.right.right.left = BinaryTree(28)
tree.right.right.right = None

print(countLeafNodes(tree))