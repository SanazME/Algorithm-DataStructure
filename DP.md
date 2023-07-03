There are two ways to implement a DP algorithm:
  1. Bottom-up, also known as tabulation.
  2. Top-down, also known as memoization.

- A bottom-up implementation's runtime is usually faster, as iteration does not have the overhead that recursion does.
- A top-down implementation is usually much easier to write. This is because with recursion, the ordering of subproblems does not matter, whereas with tabulation, we need to go through a logical ordering of solving subproblems.

- To summarize: if a problem is asking for the maximum/minimum/longest/shortest of something, the number of ways to do something, or if it is possible to reach a certain point, it is probably greedy or DP. With time and practice, it will become easier to identify which is the better approach for a given problem. Although, in general, if the problem has constraints that cause decisions to affect other decisions, such as using one element prevents the usage of other elements, then we should consider using dynamic programming to solve the problem. These two characteristics can be used to identify if a problem should be solved with DP.

**Framework to solve DP problems**
- Before we start, we need to first define a term: **state**. In a DP problem, **a state is a set of variables that can sufficiently describe a scenario**. These variables are called state variables, and we only care about relevant ones. For example, to describe every scenario in Climbing Stairs, there is only 1 relevant state variable, the current step we are on. We can denote this with an integer 
i. If i = 6, that means that we are describing the state of being on the 6th step. Every unique value of i represents a unique state.

To solve a DP problem, we need to combine 3 things:
**1. A function or data structure that will compute/contain the answer to the problem for every given state.**
- For Climbing Stairs, let's say we have an function dp where dp(i) returns the number of ways to climb to the ith step. Solving the original problem would be as easy as return dp(n).
- Typically, **top-down is implemented with a recursive function and hash map**, whereas **bottom-up is implemented with nested for loops and an array**. When designing this function or array, we also need to decide on state variables to pass as arguments. This problem is very simple, so all we need to describe a state is to know what step we are currently on i.

**2. A recurrence relation to transition between states.**
**3. Base cases, so that our recurrence relation doesn't go on infinitely.**
