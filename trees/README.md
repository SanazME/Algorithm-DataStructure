**Tree**

- Linear data structures like as arrays, linked lists, queues and stack, data is ordered and you traverse them sequentially (pic: https://medium.com/basecs/how-to-not-be-stumped-by-trees-5f36208f68a7)
- Non-linear data structure can be trees and the data traversal is not sequential. Trees are made of nodes, links. A root, parent nodes, sibling nodes (the same parent), inner nodes, leaf nodes.
- **Depth** of a node: how many edges from the node to the root 
- **Height** of a node: the maximum path from the node to the furthest leaf node
- so the height of the root is the height of the tree.
- trees are recursive data structures, a tree has a nested subtrees.
- **balanced trees**: if any two **sibling subtree** do not differ in the height by more than one level.(otherwise it will be an unbalanced tree)
- filesystems are trees

**Binary Search Tree (BST)**
1. BST, every parent node can only have two possible child nodes and not more than that
2. so the root node points to two subtrees, left subtree and right subtree and this recursively is true for each subtrees that are binary trees and two subtrees...
3. the searchable part of the BST is that all the subtrees to the left a node are smaller than the value of that node and all the subtrees to the right of that node are larger in value than the value of that node.
- like the indexing databases are BST. or **git bisect** to find where the bad commit happens uses BST.
