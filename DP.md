There are two ways to implement a DP algorithm:
  1. Bottom-up, also known as tabulation.
  2. Top-down, also known as memoization.

- A bottom-up implementation's runtime is usually faster, as iteration does not have the overhead that recursion does.
- A top-down implementation is usually much easier to write. This is because with recursion, the ordering of subproblems does not matter, whereas with tabulation, we need to go through a logical ordering of solving subproblems.

- To summarize: if a problem is asking for the maximum/minimum/longest/shortest of something, the number of ways to do something, or if it is possible to reach a certain point, it is probably greedy or DP. With time and practice, it will become easier to identify which is the better approach for a given problem. Although, in general, if the problem has constraints that cause decisions to affect other decisions, such as using one element prevents the usage of other elements, then we should consider using dynamic programming to solve the problem. These two characteristics can be used to identify if a problem should be solved with DP.

**Framework to solve DP problems**
- Before we start, we need to first define a term: **state**. In a DP problem, **a state is a set of variables that can sufficiently describe a scenario**. These variables are called state variables, and we only care about relevant ones. For example, to describe every scenario in Climbing Stairs, there is only 1 relevant state variable, the current step we are on. We can denote this with an integer 
i. If i = 6, that means that we are describing the state of being on the 6th step. Every unique value of i represents a unique state.

## To solve a DP problem, we need to combine 3 things:

**1. A function or data structure that will compute/contain the answer to the problem for every given state.**
- For Climbing Stairs, let's say we have an function dp where dp(i) returns the number of ways to climb to the ith step. Solving the original problem would be as easy as return dp(n).
- Typically, **top-down is implemented with a recursive function and hash map**, whereas **bottom-up is implemented with nested for loops and an array**. When designing this function or array, we also need to decide on state variables to pass as arguments. This problem is very simple, so all we need to describe a state is to know what step we are currently on i.

**2. A recurrence relation to transition between states.**

**3. Base cases, so that our recurrence relation doesn't go on infinitely.**

### For House Robber problem:
1. A function or array that answers the problem for a given state:
  - First, we need to decide on state variables. As a reminder, state variables should be fully capable of describing a scenario. Imagine if you had this scenario in real life - you're a robber and you have a lineup of houses. If you are at one of the houses, the only variable you would need to describe your situation is an integer - the index of the house you are currently at.
  - If the problem had an added constraint such as "you are only allowed to rob up to k houses", then k would be another necessary state variable. This is because being at, say house 4 with 3 robberies left is different than being at house 4 with 5 robberies left.

  - The problem is asking for "the maximum amount of money you can rob". Therefore, we would use either a function dp(i) that returns the maximum amount of money you can rob up to and including house i, or an array dp where dp[i] represents the maximum amount of money you can rob up to and including house i.

## 1. Min Cost Climbing Stairs
- https://leetcode.com/problems/min-cost-climbing-stairs/description/

### Options to select correct function for stat variable:
1. `dp[i]`: min cost to climb to the top **starting** from index i.
  - Assuming we have n staircase labeled from 0 to n - 1 and assuming the top is n, then dp[n] = 0, marking that if you are at the top, the cost is 0.
```py
dp[i] = min(cost[i] + dp[i + 1], cost[i] + dp[i + 2]) = cost[i] + min(dp[i + 1], dp[i + 2])`
dp[n] = 0
dp[n - 1] = cost[n - 1]

answer: min(dp[0], dp[1]) # because we can start from index 0 or index 1.
```

2. `dp[i]`: min cost to reach to step i.
```py
dp[i] = min(cost[i - 1] + dp[i - 1], cost[i - 2] + dp [i - 2])
dp[0] = 0
dp[1] = 0 # because we start from either 0 or 1
```

```py
def minCostClimbingStairs(self, cost):
    """
    :type cost: List[int]
    :rtype: int
    """
    n = len(cost)
    dp = [None for _ in range(n + 1)]
    dp[0], dp[1] = 0, 0

    for i in range(2, n + 1):
        dp[i] = min(cost[i - 1] + dp[i - 1], cost[i - 2] + dp[i - 2])

    return dp[n]

def minCostClimbingStairs(self, cost):
      """
      :type cost: List[int]
      :rtype: int
      """
      if len(cost) == 0:
          return
      if len(cost) == 1:
          return cost[0]
      
      if len(cost) == 2:
          return min(cost[0], cost[1])

      n = len(cost)
      dp = [None for _ in range(n + 1)]
      dp[n] = 0
      dp[n - 1] = cost[n - 1]
      
      for i in range(n - 2, -1, -1):
          dp[i] = cost[i] + min(dp[i + 2], dp[i + 1])

      return min(dp[0], dp[1])
```

## 2. Delete and Earn
- https://leetcode.com/problems/delete-and-earn
- We want `maxPoints(num)` to return the maximum points that we can gain if we only consider all the elements in `nums` with values between 0 and `num`.

- Then it comes to x, we have to make a choice: **take, or don't take.**

1. If we take x, then we gain points equal to x times the number of times x occurs in nums - we can pre-compute these values. For now, let's call this value gain. However, because of the deletion, by taking x, we are no longer allowed to take x - 1. The largest number that we can still consider is x - 2. Therefore, if we choose to take x, then the most points that we can have here is gain + maxPoints(x - 2), where gain is how many points we gain from taking x and maxPoints(x - 2) is the maximum number of points we can obtain from the numbers between x - 2 and 0.

2. If we choose not to take x, then we don't gain any points here, but we still may have accumulated some points from numbers smaller than x. Because we didn't take x, we did not close the door to x - 1. In this case, the most points we can have here is maxPoints(x - 1).

```py
def deleteAndEarn(self, nums: List[int]) -> int:
        
        points = defaultdict(int)
        maxVal = 0
        for num in nums:
            points[num] += num
            maxVal = max(maxVal, num)
            
        
        # maxPoints[num] = max(num + maxPoints[num - 2], maxPoints[num - 1])
        maxPoints = [0 for _ in range(maxVal + 1)]
        maxPoints[0] = 0  # the points from num = 0 is always 0
        maxPoints[1] = points[1]  # if num=1 exists in points
        
        for num in range(2, maxVal + 1):
            maxPoints[num] = max(points[num] + maxPoints[num - 2], maxPoints[num - 1])
            
        return maxPoints[maxVal]
```
