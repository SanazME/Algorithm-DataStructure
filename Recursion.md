## Recursion
- Reverse a string (try with and without recursion): https://leetcode.com/problems/reverse-string/


## Time Complexity - Recursion
- Given a recursion algorithm, its time complexity **O(T)** is typically the product of **the number of recursion invocations** (denoted as R) and **the time complexity of calculation** (denoted as O(s)) that incurs along with each recursion call:

**O(T)= R * O(s)**

- For recursive functions, it is rarely the case that the number of recursion calls happens to be linear to the size of input. For example, one might recall the example of Fibonacci number that we discussed in the previous chapter, whose recurrence relation is defined as f(n) = f(n-1) + f(n-2). At first glance, it does not seem straightforward to calculate the number of recursion invocations during the execution of the Fibonacci function.
- In this case, it is better resort to the **execution tree**, which is a tree that **is used to denote the execution flow of a recursive function** in particular. **Each node in the tree represents an invocation of the recursive function. Therefore, the total number of nodes in the tree corresponds to the number of recursion calls during the execution.**
- The execution tree of a recursive function would form an **n-ary tree, with n as the number of times recursion appears in the recurrence relation**. For instance, the execution of the Fibonacci function would form a **binary tree**. In a full binary tree with n levels, the **total number of nodes would be 2^n-1** Therefore, the upper bound (though not tight) for the number of recursion in f(n) would be **2^n -1** as well. As a result, we can estimate that the time complexity for f(n) would be **O(2^n)**.
- **Memoization not only optimizes the time complexity of algorithm, but also simplifies the calculation of time complexity.**

## Space Complexity - Recursion**
- There are mainly two parts of the space consumption that one should bear in mind when calculating the space complexity of a recursive algorithm: **recursion related and non-recursion related space.**
- The recursion related space refers to the memory cost that is incurred directly by the recursion, i.e. the stack to keep track of recursive function calls. In order to complete a typical function call, **the system allocates some space in the stack to hold three important pieces of information:

  **1. The returning address of the function call. Once the function call is completed, the program must know where to return to, i.e. the line of code after the function call.**
  **2. The parameters that are passed to the function call. **
  **3. The local variables within the function call.**
This space in the stack is the minimal cost that is incurred during a function call. However, once the function call is done, this space is freed. 

- For recursive algorithms, the function calls chain up successively until they reach a base case (a.k.a. bottom case). This implies that the space that is used for each function call is accumulated.

- **For a recursive algorithm, if there is no other memory consumption, then this recursion incurred space will be the space upper-bound of the algorithm.**
- **It is due to recursion-related space consumption that sometimes one might run into a situation called stack overflow, where the stack allocated for a program reaches its maximum space limit and the program crashes. Therefore, when designing a recursive algorithm, one should carefully check if there is a possibility of stack overflow when the input scales up**.

- For non-recursion related space, we should take into account the space cost incurred by the **memoization**.
