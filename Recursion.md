## Recursion
- Reverse a string (try with and without recursion): https://leetcode.com/problems/reverse-string/


## Time Complexity - Recursion
- Given a recursion algorithm, its time complexity **O(T)** is typically the product of **the number of recursion invocations** (denoted as R) and **the time complexity of calculation** (denoted as O(s)) that incurs along with each recursion call:

**O(T)= R * O(s)**

- For recursive functions, it is rarely the case that the number of recursion calls happens to be linear to the size of input. For example, one might recall the example of Fibonacci number that we discussed in the previous chapter, whose recurrence relation is defined as f(n) = f(n-1) + f(n-2). At first glance, it does not seem straightforward to calculate the number of recursion invocations during the execution of the Fibonacci function.
- In this case, it is better resort to the **execution tree**, which is a tree that **is used to denote the execution flow of a recursive function** in particular. **Each node in the tree represents an invocation of the recursive function. Therefore, the total number of nodes in the tree corresponds to the number of recursion calls during the execution.**
- The execution tree of a recursive function would form an **n-ary tree, with n as the number of times recursion appears in the recurrence relation**. For instance, the execution of the Fibonacci function would form a **binary tree**. In a full binary tree with n levels, the **total number of nodes would be 2^n-1** Therefore, the upper bound (though not tight) for the number of recursion in f(n) would be **2^n -1** as well. As a result, we can estimate that the time complexity for f(n) would be **O(2^n)**.
- **Memoization not only optimizes the time complexity of algorithm, but also simplifies the calculation of time complexity.**

## Space Complexity - Recursion
- There are mainly two parts of the space consumption that one should bear in mind when calculating the space complexity of a recursive algorithm: **recursion related and non-recursion related space.**
- The recursion related space refers to the memory cost that is incurred directly by the recursion, i.e. the stack to keep track of recursive function calls. In order to complete a typical function call, **the system allocates some space in the stack to hold three important pieces of information:**

  - **1. The returning address of the function call. Once the function call is completed, the program must know where to return to, i.e. the line of code after the function call.**
  - **2. The parameters that are passed to the function call.**
  - **3. The local variables within the function call.**
This space in the stack is the minimal cost that is incurred during a function call. However, once the function call is done, this space is freed. 

- For recursive algorithms, the function calls chain up successively until they reach a base case (a.k.a. bottom case). This implies that the space that is used for each function call is accumulated.

- **For a recursive algorithm, if there is no other memory consumption, then this recursion incurred space will be the space upper-bound of the algorithm.**
- **It is due to recursion-related space consumption that sometimes one might run into a situation called stack overflow, where the stack allocated for a program reaches its maximum space limit and the program crashes. Therefore, when designing a recursive algorithm, one should carefully check if there is a possibility of stack overflow when the input scales up**.

- For non-recursion related space, we should take into account the space cost incurred by the **memoization**.

## Tail recursion
- Tail recursion is a recursion where the **recursive call is the final instruction in the recursion function**. And there should be only one recursive call in the function. **There are no computations after the recursive call returned.** python and Java do not support tail recursion optimization.
- **The benefit of having tail recursion is that it could avoid the accumulation of stack overheads during the recursive calls, since the system could reuse a fixed amount space in the stack for each recursive call.**
- **Note that in tail recursion, we know that as soon as we return from the recursive call we are going to immediately return as well, so we can skip the entire chain of recursive calls returning and return straight to the original caller. That means we don't need a call stack at all for all of the recursive calls, which saves us space.**

## Pow(x,n): https://leetcode.com/problems/powx-n/
- Time complexity: **O(logN)**
- Space complexity: **O(logN)**
```py
def myPow(self, x, n):
   """
   :type x: float
   :type n: int
   :rtype: float
   """
   if n == 0:
       return 1.0

   if n < 0:
       return self.myPow(1/x, -n) 

   else:
       lower = self.myPow(x, n//2)

       if n%2 == 0:
           return lower*lower
       else:
           return x * lower * lower

```

### Merge two sorted lists:
- https://leetcode.com/problems/merge-two-sorted-lists/
- iterative in-place and recursive:
```py
def mergeTwoLists(self, l1, l2):
     """
     :type l1: ListNode
     :type l2: ListNode
     :rtype: ListNode
     """
     # Iterative - In-place
     if not l1 and not l2:
         return None
     if not l1:
         return l2
     if not l2:
         return l1

     head = dummy = TreeNode(-101)

     while l1 and l2:
         if l1.val <= l2.val:
             head.next = l1
             head = head.next
             l1 = l1.next
         else:
             head.next = l2
             head = head.next
             l2 = l2.next

     while l1:
         head.next = l1
         head = head.next
         l1 = l1.next

     while l2:
         head.next = l2
         head = head.next
         l2 = l2.next

     return dummy.next

     # Recursive
     if not l1 or not l2:
         return l1 or l2

     if l1.val <= l2.val:
         l1.next = self.mergeTwoLists(l1.next, l2)
         return l1

     else:
         l2.next = self.mergeTwoLists(l1, l2.next)
         return l2
```

### How many unique BST can be created from an integer n? 
- (https://leetcode.com/explore/learn/card/recursion-i/253/conclusion/2384/)
- If we write down the possibilities for each 1, ..., n as a root, the number of possibilities can be calculated from Catalan Number. Dynamic programming to return Catalan number:
```py
def catalanNumber(n):
    catalan = [0] * (n+1)

    def helper(n):
        if n == 0 or n == 1:
            return 1

        if catalan[n]: # non-zero value
            return catalan[n]
        else:
            for i in range(n):
                catalan[n] +=helper(i)*helper(n-1-i)
            return catalan[n]
    
    return helper(n)

```

### Unique binary search trees
- Given an integer n, return all the structurally unique BST's (binary search trees), which has exactly n nodes of unique values from 1 to n. Return the answer in any order.

- great explanation:https://leetcode.com/problems/unique-binary-search-trees-ii/discuss/929000/Recursive-solution-long-explanation
```py
def generateTrees(self, n):
    """
    :type n: int
    :rtype: List[TreeNode]
    """
    if n==1:
        return [TreeNode(n)]

    return self.helper(1, n)
            
def helper(self, start, end):
    if start > end: # edge case, see exposition below
        return [None] 

    all_trees = [] # list of all unique BSTs
    for curRootVal in range(start, end+1): # generate all roots using list [start, end]
  # recursively get list of subtrees less than curRoot (a BST must have left subtrees less than the root)
        all_left_subtrees = self.helper(start, curRootVal-1)

  # recursively get list of subtrees greater than curRoot (a BST must have right subtrees greater than the root)
        all_right_subtrees = self.helper(curRootVal+1, end) 

        for left_subtree in all_left_subtrees:   # get each possible left subtree
            for right_subtree in all_right_subtrees: # get each possible right subtree
                # create root node with each combination of left and right subtrees
                curRoot = TreeNode(curRootVal) 
                curRoot.left = left_subtree
                curRoot.right = right_subtree

      # curRoot is now the root of a BST
                all_trees.append(curRoot)

    return all_trees

 ```

## Divide and Conquer (D&C) algorithms
**1. Merge Sort**

**2. Quick sort**

### Merge Sort
- There are two approaches to implement the merge sort algorithm: top down or bottom up. In top down approach, the merge sort algorithm can be divided into 3 steps like D&C algorithms:
    1. Divide the given unsorted list into several sublists.  (Divide)

    2. Sort each of the sublists recursively.  (Conquer)

    3. Merge the sorted sublists to produce new sorted list.  (Combine)

- The recursion in step (2) would reach the base case where the input list is either empty or contains a single element. Now, we have reduced the problem down to a merge problem, which is much simpler to solve. Merging two sorted lists can be done in linear time complexity {O(N)}O(N), where {N}N is the total lengths of the two lists to merge.

- **Complexity:**
- The overall **time complexity** of the merge sort algorithm is **O(NlogN)**, where N is the length of the input list. To calculate the complexity, we break it down to the following steps:

  1. We recursively divide the input list into two sublists, until a sublist with single element remains. This dividing step computes the midpoint of each of the sublists, which takes O(1) time. This step is repeated N times until a single element remains, therefore the total time complexity is **O(N).**
 
  2. Then, we repetitively merge the sublists, until one single list remains. The recursion tree in Fig. 1 or Fig. 2 above is useful for visualizing how the recurrence is iterated. As shown in the recursion tree, there are a total of N elements on each level. Therefore, it takes **O(N) time for the merging process to complete on each level**. And since there are a total of **logN levels**, the **overall complexity of the merge process is** **O(NlogN).**
Taking into account the complexity of the above two parts in the merge sort algorithm, we conclude that the overall time complexity of merge sort is O(N\log{N})O(NlogN).

The **space complexity** of the merge sort algorithm is **O(N)**, where N is the length of the input list, since we need to keep the sublists as well as the buffer to hold the merge results at each round of merge process.

```py
def sortArray(self, nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    if len(nums) <= 1:
        return nums

    pivot = len(nums)//2
    left_side = self.sortArray(nums[0:pivot])
    right_side = self.sortArray(nums[pivot:])

    return self.merge(left_side, right_side)


def merge(self, left_side, right_side):

    left, right = 0, 0
    result = []

    while left < len(left_side) and right < len(right_side):
        if left_side[left] <= right_side[right]:
            result.append(left_side[left])
            left += 1
        else:
            result.append(right_side[right])
            right += 1

    if left < len(left_side):
        result.extend(left_side[left:])
    if right < len(right_side):
        result.extend(right_side[right:])

    return result
```

### Quick sort
- Picture: https://leetcode.com/explore/learn/card/recursion-ii/470/divide-and-conquer/2870/
- In detail, given a list of values to sort, the quick sort algorithm works in the following steps:

1. First, it selects a value from the list, which serves as a **pivot** value to divide the list into two sublists. One sublist contains all the values that are less than the pivot value, while the other sublist contains the values that are greater than or equal to the pivot value. This process is also called **partitioning**. The strategy of choosing a pivot value can vary. Typically, one can choose the first element in the list as the pivot, or randomly pick an element from the list.

2. After the partitioning process, the original list is then reduced into two smaller sublists. We then **recursively** sort the two sublists.

3. After the partitioning process, we are sure that all elements in one sublist are less or equal than any element in another sublist. Therefore, we can simply **concatenate** the two sorted sublists that we obtain in step [2] to obtain the final sorted list. 

```py
def quicksort(lst):
    """
    Sorts an array in the ascending order in O(n log n) time
    :param nums: a list of numbers
    :return: the sorted list
    """
    n = len(lst)
    qsort(lst, 0, n - 1)
    return lst

def qsort(lst, lo, hi):
    """
    Helper
    :param lst: the list to sort
    :param lo:  the index of the first element in the list
    :param hi:  the index of the last element in the list
    :return: the sorted list
    """
    if lo < hi:
        p = partition(lst, lo, hi)
        qsort(lst, lo, p - 1)
        qsort(lst, p + 1, hi)

def partition(lst, lo, hi):
    """
    Picks the last element hi as a pivot
     and returns the index of pivot value in the sorted array
    """
    pivot = lst[hi]
    i = lo
    for j in range(lo, hi):
        if lst[j] < pivot:
            lst[i], lst[j] = lst[j], lst[i]
            i += 1
    lst[i], lst[hi] = lst[hi], lst[i]
    return i

print(quicksort([1,5,3,2,8,7,6,4]))
```

## Master Theorem
- https://leetcode.com/explore/learn/card/recursion-ii/470/divide-and-conquer/2871/


## Backtracking


### Subset sum problem
- https://www.youtube.com/watch?v=34l1kTIQCIA


### Partial subset sum
- https://leetcode.com/problems/partition-equal-subset-sum/
- https://leetcode.com/problems/partition-equal-subset-sum/discuss/462699/Whiteboard-Editorial.-All-Approaches-explained.
- Note that sum of all possible subsets of a list can be found by O(2^n)? time when we use tree:
`list=[1,2,3]`
```
                          0
                       /    \ +1
                      0      1
                     /        \ +2
                    0,1       2,3
                   /            \ +3
                  0,1,2,3       3,4,5,6
                  
              => 0,1,2,3,4,5,6
```                  
```py
def canPartition(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
#         https://leetcode.com/problems/partition-equal-subset-sum/discuss/462699/Whiteboard-Editorial.-All-Approaches-explained.
        """
        Brute force: search for a subset S1 with sum(S1)= sum(nums)/2 
        - search for every possible sum of all possible subsets of nums
        time complexity?????? O(2^n) but with memoization DP 
        """
        totalSum = sum(nums)
        if (totalSum % 2 != 0):
            return False
        
        targetSum = totalSum/2
        subsetSums = set([0])
        
        for _, num in enumerate(nums):
            ll = []
            for ele in subsetSums:
                if ele+num == targetSum:
                    return True
                ll.append(ele+num)
            subsetSums.update(ll)
            
        if targetSum in subsetSums:
            return True
        else:
            return False
```
### Knapsack solution for the above problem:
- https://leetcode.com/problems/partition-equal-subset-sum/discuss/462699/Whiteboard-Editorial.-All-Approaches-explained.
- The highest value I get from the first `i` items having a weight constraint of `w`:
```
The highest value 
out of all possible
combinations of the first i elements = max(If I have the highest value from the first i-1 elements, I can either ignore the ith element or choose it)
with a weight containt of w


V[i][w] = max(V[i-1][w] , V[i-1][w - wi] + Vi)
- Create 2D table with rows as elements and columns as weight containt
- if w - wi < 0, V[i][w] = V[i-1][w]
```
- The similarity of this problem and Knapsack one:
```
                            My problem    |  Knapsack
                         --------------------------------
   - Given n number (array)               |   - n items, their weights
   - Constraint: A target sum: sum A / 2  |   - a weight constaint K
   --------------------------------------------------------------------------
   I want to find out IF there is  a      |   Maximum value of my ideam combination
   combination that sums up to sumA/2     |
   

- if there is combination out if the first i element that sums up to w, it can be found either by excluding the ith element(if we find a combination of the first i-1 elements that sums up to w) or including the ith element(if we find a combination of the first i-1 elements that sums up to w-wi and so by adding ith element, the sum goes up to w)
    B[i][w] = B[i-1][w] || B[i-1][w-wi]

```

- Not optimized: Time O(nW) and space O(nW) where W is the sum of elements
```py
def canPartition(nums):
    sumAll = sum(nums)
    if sumAll % 2 != 0:
        return False
    
    target = sumAll/2
    # Sum of the first i items (w):
    #  B[i][w] = B[i-1][w] || B[i-1][w-wi]
    # 
    # Create Rows & columns
    arr = [[False]*(target+1) for i in range(len(nums)+1)]
    arr[0][0] = True
    
    for i in range(1,len(nums)+1):
        for j in range(target+1):
            if j >= nums[i-1]:
                arr[i][j] = arr[i-1][j] | arr[i-1][j-nums[i-1]]
            else:
                arr[i][j] = arr[i-1][j]
    
    return arr[len(nums)][target]

```
- Optimized in space Knapsack version O(W). To calculate `B[i][w]`, we just need the element above and an item before that. So we update our elements from right to left, we only need one row instead of the whole 2D array:
```py

def canPartition(nums):
    sumAll = sum(nums)
    if sumAll % 2 != 0:
        return False
    
    target = sumAll/2
    # Sum of the first i items (w):
    #  B[i][w] = B[i-1][w] || B[i-1][w-wi]
    # 
    # Create Rows & columns
    arr = [False]*(target + 1)
    arr[0] = True
    
    for i in range(len(nums)):
        for j in range(target, 0, -1):
            if j >= nums[i-1]:
                arr[j] = arr[j] | arr[j-nums[i-1]]
            
    return arr[target]
```

- https://leetcode.com/problems/subsets-ii/
-
**BackTracking**
When designing our recursive function, there are two main points that we need to consider at each function call:

- Whether the element under consideration has duplicates or not.
- If the element has duplicates, which element among the duplicates should be considered while creating a subset.
 
![image](https://user-images.githubusercontent.com/5471080/180220507-9ed54e3a-b527-43d7-b9aa-487033142ce6.png)

The recursion tree illustrating how distinct subsets are created at each function call.Here the numbers in blue indicate the starting position of the nums array where we should start scanning at that function call.

The above illustration gives us a rough idea about how we get the solution in a backtracking manner. Note that the order of the subsets in the result is the preorder traversal of the recursion tree. All that is left to do is to code the solution.

Start with an empty list and the starting index set to 0. At each function call, add a new subset to the output list of subsets. Scan through all the elements in the nums array from the starting index (written in blue in the above diagram) to the end. Consider one element at a time and decide whether to keep it or not. If we haven't seen the current element before, then add it to the current list and make a recursive function call with the starting index incremented by one. Otherwise, the subset is a duplicate and so we ignore it. Thus, if in a particular function call we scan through k distinct elements, there will be k different branches.

**Algorithm**
1. First, sort the array in ascending order.
2. Use a recursive helper function helper to generate all possible subsets. The helper has the following parameters:

  - Output list of subsets (subsets).
  - Current subset (currentSubset).
  - nums array.
  - the index in the nums array from where we should start scanning elements at that function call (index).

3. At each recursive function call:

  - Add the currentSubset to the subsets list.
  - Iterate over the nums array from index to the array end.

    - If the element is considered for the first time in that function call, add it to the currentSubset list. Make a function call to helper with index = current element position + 1.
    - Otherwise, the element is a duplicate. So we skip it as it will generate duplicate subsets (refer to the figure above).
    - While backtracking, remove the last added element from the currentSubset and continue the iteration.
4. Return `subsets` list.

```py
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        
        nums.sort()
        subsets = []
        
        def helper(currentSubset, idx):
            # Add the subset formed so far to the subsets list
            subsets.append(currentSubset[:]) # it is a shallow copy
            
            for i in range(idx, len(nums)):
                # if the current element is a duplicate, ignore
                if i != idx and nums[i] == nums[i-1]:
                    continue
                    
                currentSubset.append(nums[i])
                helper(currentSubset, i + 1)
                currentSubset.pop()
            
        
        helper([], 0)
        
        return subsets

```
- Time complexity: `O(n.2^n)`
As we can see in the diagram above, this approach does not generate any duplicate subsets. Thus, in the worst case (array consists of nn distinct elements), the total number of recursive function calls will be 2 ^ n. Also, at each function call, a deep copy of the subset currentSubset generated so far is created and added to the subsets list. This will incur an additional O(n)O(n) time (as the maximum number of elements in the currentSubset will be nn). So overall, the time complexity of this approach will be O(n⋅2^n ).

- https://leetcode.com/problems/longest-palindromic-substring/
```py
class Solution(object):
    def __init__(self):
        self.dic = {}
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """           
        if len(s) <= 1:
            return s
    
        soFar = ''
        for i, char in enumerate(s):
            # for odd case: 'aba'
            tmp = self.helper(s, i, i)
            if len(tmp) >= len(soFar):
                soFar = tmp

            # for even case: 'abba'
            tmp = self.helper(s, i, i+1)

            if len(tmp) >= len(soFar):
                soFar = tmp

        return soFar
        
        
    def helper(self, s, left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left+1:right]
```
