https://www.geeksforgeeks.org/binary-tree-set-3-types-of-binary-tree/

**Binary Tree **| Set 3 (Types of Binary Tree)

We have discussed Introduction to Binary Tree in set 1 and Properties of Binary Tree in Set 2. In this post, common types of binary is discussed.

Following are common types of Binary Trees.

Full Binary Tree A Binary Tree is full if every node has 0 or 2 children. Following are examples of a full binary tree. We can also say a full binary tree is a binary tree in which all nodes except leaves have two children.

               18
           /       \
         15         30
        /  \        /  \
      40    50    100   40

             18
           /    \
         15     20
        /  \
      40    50
    /   \
   30   50

               18
            /     \
          40       30
                   /  \
                 100   40


In a Full Binary, number of leaf nodes is number of internal nodes plus 1
       L = I + 1
Where L = Number of leaf nodes, I = Number of internal nodes
See Handshaking Lemma and Tree for proof.



Complete Binary Tree: A Binary Tree is complete Binary Tree if all levels are completely filled except possibly the last level and the last level has all keys as left as possible

Following are examples of Complete Binary Trees

               18
           /       \
         15         30
        /  \        /  \
      40    50    100   40


               18
           /       \
         15         30
        /  \        /  \
      40    50    100   40
     /  \   /
    8   7  9
Practical example of Complete Binary Tree is Binary Heap.



Perfect Binary Tree A Binary tree is Perfect Binary Tree in which all internal nodes have two children and all leaves are at the same level.
Following are examples of Perfect Binary Trees.

               18
           /       \
         15         30
        /  \        /  \
      40    50    100   40


               18
           /       \
         15         30
A Perfect Binary Tree of height h (where height is the number of nodes on the path from the root to leaf) has 2^h – 1 node.

Example of a Perfect binary tree is ancestors in the family. Keep a person at root, parents as children, parents of parents as their children.



Balanced Binary Tree
A binary tree is balanced if the height of the tree is O(Log n) where n is the number of nodes.
A balanced binary tree has roughly the same number of nodes in the left and right subtrees of the root.
For Example, AVL tree maintains O(Log n) height by making sure that the difference between heights of left and right subtrees is 1. Red-Black trees maintain O(Log n) height by making sure that the number of Black nodes on every root to leaf paths are same and there are no adjacent red nodes. Balanced Binary Search trees are performance wise good as they provide O(log n) time for search, insert and delete.



A degenerate (or pathological) tree A Tree where every internal node has one child. Such trees are performance-wise same as linked list.

      10
      /
    20
     \
     30
      \
      40


** Binary Heap **
- Balanced binary tree O(logn) for insertion, deletion and search - complete binary trees

In order to guarantee logarithmic performance, we must keep our tree balanced. A balanced binary tree has roughly the same number of nodes in the left and right subtrees of the root. In our heap implementation we keep the tree balanced by creating a complete binary tree. A complete binary tree is a tree in which each level has all of its nodes. The exception to this is the bottom level of the tree, which we fill in from left to right. This diagram shows an example of a complete binary tree:

Binary Heap
https://www.geeksforgeeks.org/binary-heap/
A Binary Heap is a Binary Tree with following properties.
1) It’s a complete tree (All levels are completely filled except possibly the last level and the last level has all keys as left as possible). This property of Binary Heap makes them suitable to be stored in an array.

2) A Binary Heap is either Min Heap or Max Heap. In a Min Binary Heap, the key at root must be minimum among all keys present in Binary Heap. The same property must be recursively true for all nodes in Binary Tree. Max Binary Heap is similar to MinHeap.

Examples of Min Heap:



            10                      10
         /      \               /       \
       20        100          15         30
      /                      /  \        /  \
    30                     40    50    100   40
How is Binary Heap represented?
A Binary Heap is a Complete Binary Tree. A binary heap is typically represented as an array.

The root element will be at Arr[0].
Below table shows indexes of other nodes for the ith node, i.e., Arr[i]:
Arr[(i-1)/2]	Returns the parent node
Arr[(2*i)+1]	Returns the left child node
Arr[(2*i)+2]	Returns the right child node
The traversal method use to achieve Array representation is Level Order

Applications of Heaps:
1) Heap Sort: Heap Sort uses Binary Heap to sort an array in O(nLogn) time and
space complexity O(1).

algorithm       time          space
----------    --------      ----------
Heapsort     O(nlogn)        O(1)
Quicksort    O(nlogn)        O(logn)
Mergesort    O(nlogn)       O(n)



2) Priority Queue: Priority queues can be efficiently implemented using Binary Heap because it supports insert(), delete() and extractmax(), decreaseKey() operations in O(logn) time. Binomoial Heap and Fibonacci Heap are variations of Binary Heap. These variations perform union also efficiently.

3) Graph Algorithms: The priority queues are especially used in Graph Algorithms like Dijkstra’s Shortest Path and Prim’s Minimum Spanning Tree.

4) Many problems can be efficiently solved using Heaps. See following for example.
a) K’th Largest Element in an array.
b) Sort an almost sorted array/
c) Merge K Sorted Arrays.


### Priority queques:

- we use binary heap data structure for priority queques.
- with binary heap, dequeue and enqueue are O(logn), instead of inserting into a list or arrus O(n) and sorting a list O(nlogn)
- when we diagram the heap it looks a lot like a tree, but when we implement it we use only a single list as an internal representation. The binary heap has two common variations: the **min heap**, in which the smallest key is always at the front, and the **max heap**, in which the largest key value is always at the front.
- we can build a binary heap from a list of keys in O(n). We can use a sorting algorithm that uses the heap and sorts a list in O(nlogn). The binary heap is a balanced tree which means that the tree has a height of O(logn) where n is the number of nodes. The balanced tree provides O(logn) time for search, insert and delete.

** Binary heap operations:**

- **BinaryHeap()** creates a new, empty, binary heap.
- **insert(k)** adds a new item to the heap.
- **findMin() **returns the item with the minimum key value, leaving item in the heap.
- **delMin()** returns the item with the minimum key value, removing the item from the heap.
- **isEmpty()** returns true if the heap is empty, false otherwise.
- **size()** returns the number of items in the heap.
- **buildHeap(list)** builds a new heap from a list of keys.

More in algorithms_problems/tree

"""
Binary Search Tree (BST)
"""
The property of BST is that the keys less than the parent are on the left and the keys more than the parent are on the right side of the parent node and both left and right subtrees are BSTs.

insertion, deletion and search in a BST is better than BT. It is O(logn), the height of the BST.

Depth-first travesals in a binary tree are: Preorder (NLR), Inorder(LNR) and Postorder (LRN). All of these travesals first go to the depth of the tree before exploring the breath of the tree. 
BT level order traversal is a breath-first traversal.a

To visit all nodes in a sorted order -> In order traversal.

"""
map ADT methods
"""
Notice that this interface is very similar to the Python dictionary.

1**Map()**          Create a new, empty map.
2**put(key, val)**  Add a new key-value pair to the map. If the key is already in the map then replace the old value with the new value.
3**get(key)**       Given a key, return the value stored in the map or None otherwise.
4**del**            Delete the key-value pair from the map using a statement of the form **del map[key]**.
5**len()**          Return the number of key-value pairs stored in the map.
6**in Return True** for a statement of the form key in map, if the given key is in the map.

### Summary of Map ADT Implementations ###

Over the past two chapters we have looked at several data structures that can be used to 
implement the map abstract data type. 
A binary Search on a list, 
a hash table, 
a binary search tree, and 
a balanced binary search tree. 
To conclude this section, let’s summarize the performance of each data structure for the key operations defined by the map ADT (see Table 1).

| operation | sorted list   | hash table | BST   |  AVL   |
|-----------|---------------|------------|-------|--------|
|   put     |    O(n)       |  O(1)      |  O(n) | O(logn)|
|   get     |    O(logn)    |  O(1)      |  O(n) | O(logn)|
|   in      |    O(logn)    |  O(1)      |  O(n) | O(logn)|
|   del     |    O(n)       |  O(1)      |  O(n) | O(logn)|


* A binary tree for parsing and evaluating expressions.
* A binary tree for implementing the map ADT.
* A balanced binary tree (AVL tree) for implementing the map ADT.
* A binary tree to implement a min heap.
* A min heap used to implement a priority queue.