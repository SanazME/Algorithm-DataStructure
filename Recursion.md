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

## Merge two sorted lists:
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

- How many unique BST can be created from an integer n? (https://leetcode.com/explore/learn/card/recursion-i/253/conclusion/2384/)
- If we write down the possibilities for each 1, ..., n as a root, the number of possibilities can be calculated from Catalan Number. Dynamic programming to return Catalan number:
```py
def catalanNumber(n):
    catalan = [0] * (n+1)

    def helper(n):
        if n == 0 or n == 1:
            return 1

        if catalan[n]: # non-zero value
            return catalan[n
        else:
            catalan[n] = 0
            for i in range(n):
                catalan[n] +=helper(i)*helper(n-1-i)
            return catalan[n]
    
    return helper(n)

```
