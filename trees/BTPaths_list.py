"""
Given a binary tree, return all root-to-leaf paths.

Note: A leaf is a node with no children.

Example:

Input:

   1
 /   \
2     3
 \
  5

Output: ["1->2->5", "1->3"]

Explanation: All root-to-leaf paths are: 1->2->5, 1->3
"""

# Solution 1 - slower
#lower level recursion returns list, not a single value (i.e. += vs. .append())
#values in the list returned by the lower level recursion call should be prepended with "root->"#

def all_paths(root):
    if not root:
        return []
    else:
        if isLeaf(root):
            return [str(root.value)]
        else:
            left_tree = all_paths(root.left)
            right_tree = all_paths(root.right)
            
            list1=[]
            for leaf in (left_tree+right_tree):
                list1.append(str(root.value)+'->'+leaf)
            return list1

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

print(all_paths(tree))

# Second solution - faster without iteration on the list
def all_paths_2(root):
    if not root:
        return []
    else:
        result=[]
        
        ## Helper function
        
        def helper(root, pathString):
            if isLeaf(root):
                if len(pathString)==0:
                    result.append(str(root.value))
                else:
                    result.append(pathString[2:]+'->'+str(root.value))                
                return
                    
            if root.left:
                helper(root.left, pathString+"->"+str(root.value))
            if root.right:
                helper(root.right, pathString+"->"+str(root.value))        
        ###
        
        helper(root, "")
        return result

print(all_paths_2(tree))