"""
How are all leaves of a binary search tree printed

"""
class BinaryTree(object):
    def __init__(self,value):
        self.value = value
        self.left = None
        self.right = None

def isLeaf(node):
    return (not node.left) and (not node.right)

def leafNodes(root):
    if root is None:
        return
    else:
        leafNodes(root.left)
        leafNodes(root.right)
        if isLeaf(root):
            print(root.value)

tree = BinaryTree(10)
tree.left = BinaryTree(0)
tree.right = BinaryTree(25)

tree.left.left = BinaryTree(-1)
tree.left.right = None

tree.right.left = BinaryTree(16)
tree.right.right = BinaryTree(32)

tree.right.right.left = BinaryTree(28)
tree.right.right.right = None

leafNodes(tree)

