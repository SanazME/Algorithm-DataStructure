"""
If you look at this method you will find that we are examining leaves before examining root. We start the post order traversal from the root by pushing it into a Stack and then loop until out Stack is empty. At each iteration, we peek() the element from Stack i.e. retrieve it without removing and check if it's a leaf, if yes then we pop() the element and print its value, which means the node is visited.

If it's not a leaf then we check if it has a right node, if yes we store into a tree and set it to null, similarly, we check if it has left a node, if yes we push into the stack and then mark it null. We first insert right node because Stack is a LIFO (last in first out) data structure and as per post order traversal we need to explore left subtree before right subtree


Read more: http://www.java67.com/2017/05/binary-tree-post-order-traversal-in-java-without-recursion.html#ixzz5m0k7kYG9

"""

def postOrder(root):
    if root is None:
        return
    else:
        stack=[root]
        
        while stack:
            # peek last element
            node = stack[-1]
            
            if isLeaf(node):
                stack.pop()
                print(node.value)
            else:
                if node.right:
                    stack.append(node.right)
                    node.right=None
                if node.left:
                    stack.append(node.left)
                    node.left=None
                    
def isLeaf(node):
    if (not node.left) and (not node.right):
        return True
    else:
        return False



"""
1.1 Create an empty stack
2.1 Do following while root is not NULL
    a) Push root's right child and then root to stack.
    b) Set root as root's left child.
2.2 Pop an item from stack and set it as root.
    a) If the popped item has a right child and the right child 
       is at top of stack, then remove the right child from stack,
       push the root back and set root as root's right child.
    b) Else print root's data and set root as NULL.
2.3 Repeat steps 2.1 and 2.2 while stack is not empty.



"""
def postOrder(root):
    if root is None:
        return
    else:
        stack, result=[root],[]
        
        while stack:
            
            while root:
                if root.right:
                    stack.append(root.right)
                stack.append(root)
                root = root.left
                
            node = stack.pop()
            
            if not stack:
                return result
            
            if (node.right) and (node.right == stack[-1]):
                rightChild=stack.pop()
                stack.append(node)
                root = rightChild
            else:
                result.append(node.value)
                root=None

class BinaryTree(object):
    def __init__(self, value):
        self.value=value
        self.left=None
        self.right=None


tree = BinaryTree(10)
tree.left = BinaryTree(0)
tree.right = BinaryTree(25)

tree.left.left = BinaryTree(-1)
tree.left.right = None

tree.right.left = BinaryTree(16)
tree.right.right = BinaryTree(32)

tree.right.right.left = BinaryTree(28)
tree.right.right.right = None

postOrder(tree)