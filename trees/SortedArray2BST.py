"""
Given an array where elements are sorted in ascending order, convert it to a height balanced BST.

For this problem, a height-balanced binary tree is defined as a binary tree in which the depth of 
the two subtrees of every node never differ by more than 1.

Example:

Given the sorted array: [-10,-3,0,5,9],

One possible answer is: [0,-3,9,-10,null,5], which represents the following height balanced BST:

      0
     / \
   -3   9
   /   /
 -10  5


"""

"""
Time complexity: O(n)
"""
# Definition for a binary tree node.
class TreeNode(object):
  def __init__(self, x):
    self.val = x
    self.left = None
    self.right = None

class Solution(object):
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        if not nums:
            return None

        mid = len(nums)//2
        tree = TreeNode(nums[mid])
        tree.left = self.sortedArrayToBST(nums[:mid])
        tree.right = self.sortedArrayToBST(nums[mid+1:])
        return tree

s = Solution()
out = s.sortedArrayToBST([1, 2, 3, 4, 5, 6, 7, 12, 24, 125])

def printTree(tree):
  if tree is None:
    return

  print(tree.val)
  printTree(tree.left)
  printTree(tree.right)

printTree(out)

"""
Slicing an array is expensive:
"""
def sortedArrayToBST(nums):

    def convert(left,right):
        if left > right:
            return None
        mid = (left+right)//2
        tree = TreeNode(nums[mid])
        print(tree.val)
        tree.left = convert(left,mid-1)
        tree.right = convert(mid+1,right)
        return tree

    return convert(0,len(nums)-1)
    
