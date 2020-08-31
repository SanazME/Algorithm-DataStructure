"""
Question: You are given 3 things. The root of a binary tree, a single start node in the binary 
tree, and a number k. Return all nodes that are k "hops" away from the start node in the 
binary tree. Return a list of the values held at those nodes.

https://www.youtube.com/watch?v=nPtARJ2cYrg&list=PLiQ766zSC5jND9vxch5-zT7GuMigiWaV_

Complexities

n = total amount of nodes in the binary tree
m = total edges

Time: O( n + m )

This is "standard to Breadth First Search". We upper bound the time by the number of nodes we can visit and edges we can traverse (at maximum).

Space: O( n )

We have a hashtable upper bounded by n mappings, a mapping to each node's parent.

Solution descriptopn: It is a breath-first search and for BFS we can use a queue
and BT is a acyclic connected graph (directed graph (downwards)). But search like a graph has a problem here
we cannot go to the parent of the node. BFT (level-order traversal) on a BT, we cannot
go back to the node parent because BT is a directed graph. To solve it, we should save the parents
of each node with O(1) time access. A data structure that gives a constant time O(1)
 access to any object of interest is Hash Tables. With a hash table, we change a directed graph to
 an indirected graph.
 We can run through the tree (we know inorder, preorder, postorder traversals) and create a 
 hashtable and map each node to its parents. After that, we execute a normal BFS on the BT using a
 Queue.

 Important: A Queue makes the search a BFS. A stack makes search a DFS.
 Important

 Once we have a map of the parents of each node (map for an undirected graph), we start with the
 start node and add it to the queue and the seen (dic):
 seen:{  }
 Queue: (front)         (end)
"""
"""
We are given a binary tree (with root node root), a target node, and an integer value K.

Return a list of the values of all nodes that have a distance K from the target node.  
The answer can be returned in any order.

 LeetCode

Example 1:

Input: root = [3,5,1,6,2,0,8,null,null,7,4], target = 5, K = 2

Output: [7,4,1]

Explanation: 
The nodes that are a distance 2 from the target node (with value 5)
have values 7, 4, and 1.



Note that the inputs "root" and "target" are actually TreeNodes.
The descriptions of the inputs above are just serializations of these objects.

"""
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    def distanceK(self, root, target, K):
        from collections import deque
        """
        :type root: TreeNode
        :type target: TreeNode
        :type K: int
        :rtype: List[int]
        """
        # Map each node to its parent by a hash table
        def mapNodeToParent(node,parent):
            if node:
                node.parent = parent
                mapNodeToParent(node.left, node)
                mapNodeToParent(node.right, node)

        mapNodeToParent(root,None)
        
        queue = deque([(target,0)])
        seen={target} # Set like dictionary without value
        
        while queue:
            if queue[0][1]==K:
                return [node.val for node,level in queue]
            
            node, level = queue.popleft()
            for nei in (node.left, node.right, node.parent):
                if nei and nei not in seen:
                    seen.add(nei)
                    queue.append((nei, level+1))  
            
            
        return []        
                
    def addQueue(self,node, level, queue, seen):
        if node:
            if node not in seen:
                seen.add(node)
                queue.appendleft((node, level+1))
            
            
"""
root = [3,5,1,6,2,0,8,null,null,7,4], target = 5, K = 2

Output: [7,4,1]
"""

tree = TreeNode(3)
tree.left = TreeNode(5)
tree.right=TreeNode(1)
tree.left.left = TreeNode(6)
tree.left.right = TreeNode(2)
tree.right.left = TreeNode(0)
tree.right.right = TreeNode(8)
tree.left.left.left = None
tree.left.left.right = None
tree.left.right.left = TreeNode(7)
tree.left.right.right = TreeNode(4)

s = Solution()
print(s.distanceK(tree, tree.left,2))