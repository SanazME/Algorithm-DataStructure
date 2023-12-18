There are two ways to implement a DP algorithm:
  1. Bottom-up, also known as tabulation.
  2. Top-down, also known as memoization.

- A bottom-up implementation's runtime is usually faster, as iteration does not have the overhead that recursion does.
- A top-down implementation is usually much easier to write. This is because with recursion, the ordering of subproblems does not matter, whereas with tabulation, we need to go through a logical ordering of solving subproblems.

- To summarize: if a problem is asking for the maximum/minimum/longest/shortest of something, the number of ways to do something, or if it is possible to reach a certain point, it is probably greedy or DP. With time and practice, it will become easier to identify which is the better approach for a given problem. Although, in general, if the problem has constraints that cause decisions to affect other decisions, such as using one element prevents the usage of other elements, then we should consider using dynamic programming to solve the problem. These two characteristics can be used to identify if a problem should be solved with DP.

- Memoization is a type of caching used in top-down DP approach. It refers more specfically to **caching function that returns the value.**

- Tabulation outperforms memoization by a constant factor. This is because the tabulation has no overhead of recursion which reduces the time for resolving the call stack from the stack memory.

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


## ** 3. Maximum Score from performing multiplication operations
- https://leetcode.com/problems/maximum-score-from-performing-multiplication-operations

To determine a state, we essentially need 3 things:

1. **left**: specify we have used **left** integers from the left side of nums so far. Next, we may use **nums[left]**

2. **right**: specify we have used **right** integers from the right side of nums so far. Next, we may use **nums[right]**

3. **op**: number of operations done.

we can calculate **right** from **left** and **op**. op shows the total operations have done so far:
- `op - left` : right operations have done
- `len(nums) - (op - right)` : number of **right** elements coming from the end to the start
- `[len(nums) - (op - left)] - 1` : index of **right** 

### Top-down:
- Top-down approach TLE beacuse of recursion depth: `O(M^2)` where multipliers can vary from 0 to M - 1. Now, in two recursive calls that we are making, one time we are incrementing `left`, along with `op`. Other time, we are not incrementing `left`, but incrementing `op`. So, `left` is at most `op`. Thus, `left` also varies from 0 to `M-1`. So, there are `O(M^2)`.
- Space complexity: `O(M^2)`, the memo will store at most M^2such pairs!

```py
class Solution:
    def maximumScore(self, nums: List[int], multipliers: List[int]) -> int:

        # Number of Operations
        m = len(multipliers)

        # For Right Pointer
        n = len(nums)

        memo = {}

        def dp(op, left):
            if op == m:
                return 0

            # If already computed, return
            if (op, left) in memo:
                return memo[(op, left)]

            l = nums[left] * multipliers[op] + dp(op+1, left+1)
            r = nums[(n-1)-(op-left)] * multipliers[op] + dp(op+1, left)

            memo[(op, left)] = max(l, r)

            return memo[(op, left)]

        # Zero operation done in the beginning
        return dp(0, 0)
```

### Bottom-up:

- we only need part of the 2D array where `op >= left`. Because if `op` reaches zero there is no operations left for the remaining of `left` elements.
```py
class Solution:
    def maximumScore(self, nums: List[int], multipliers: List[int]) -> int:

        # Number of Operations
        m = len(multipliers)

        # For Right Pointer
        n = len(nums)

        dp = [[0] * (m + 1) for _ in range(m + 1)]

        for op in range(m - 1, -1, -1):
            for left in range(op, -1, -1):

                dp[op][left] = max(multipliers[op] * nums[left] + dp[op + 1][left + 1],
                                   multipliers[op] * nums[n - 1 - (op - left)] + dp[op + 1][left])

        return dp[0][0]
 
```

- Time complexity: `O(M^2)`. op varies from M-1 to 0, and left varies from op to 0. This is equivalent to iterating half matrix of order M×M. So, we are computing O(M^2/2).

- Space complexity: `O(M^2)` as evident from the dp array.

## 4. Longest Common Subsequence
- https://leetcode.com/problems/longest-common-subsequence/description/
- Finding a longest common subsequent has read-world application: for diffing file names in git.

### Bottom-up
- we first try to find the state variables and the function:
  - state variables will be i ad j to show where we are in text1 and text2: `text1[0...i], text2[0...j]`
  - the function: `dp[i][j]` : the longest common subsequent up until i and j.
  - The relation:
```py
if text1[i] == text2[j]:
    dp[i][j] = 1 + dp[i-1][j-1]  # we are moving diagonally in the table
else:
    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])  # we down and left in the table
```
- so for i = 0 and j = 0, if the letter is the same, the rest of first column or first row will be 1 as well:
```py
def longestCommonSubsequence(self, text1: str, text2: str) -> int:
    n, m = len(text1), len(text2)
    dp = [[0 for _ in range(m)] for _ in range(n)]

    # if any letter in text1 is equal to the first letter of text2 then for the rest of letters in text1, value is 1
    for i in range(n):
        if text1[i] == text2[0]:
            for ii in range(i, n):
                dp[ii][0] = 1
            break
    # if any letter in text2 is equal to the first letter of text1 then for the rest of letters in text2, value is 1
    for j in range(m):
        if text2[j] == text1[0]:
            dp[0][j:] = [1 for _ in range(m - j)]
            break

    for i in range(1, n):
        for j in range(1, m):
            if text1[i] == text2[j]:
                dp[i][j] = 1 + dp[i-1][j-1]
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[-1][-1]
```

- another way to have this table is that instead of that complicated way of setting the first column and first row to one is to start from the end of `text1` and `text2` and move up to the start:
```py
dp2 = [[0 for _ in range(m + 1)] for _ in range(n + 1)]

for i in range(n-1, -1, -1):
    for j in range(m-1, -1, -1):
        if text1[i] == text2[j]:
            dp2[i][j] = 1 + dp2[i+1][j+1]
        else:
            dp2[i][j] = max(dp2[i+1][j], dp2[i][j+1])

return dp2[0][0]
```
# DP Patterns:
- https://leetcode.com/discuss/general-discussion/458695/Dynamic-Programming-Patterns
- https://leetcode.com/discuss/interview-question/1380561/Template-For-Dynamic-programming
## 5. Minimum Path Sum
- https://leetcode.com/problems/minimum-path-sum/description/?envType=list&envId=55ac4kuc
- Bottom-up and Top-bottom:
**Solution**: https://leetcode.com/problems/minimum-path-sum/solutions/3345941/explained-in-detailed-with-image-dry-run-of-same-i-p-most-upvoted/

## 6. Word Break
- https://leetcode.com/problems/word-break/editorial/
- 

## 7. Coin change
- we can use BFS: https://github.com/SanazME/Algorithm-DataStructure/blob/master/trees/README.md#7-coin-change
- or DP:
  
**7.1 Top-bottom**
  - The helper is `F(s) = min(F(s - coin) + 1 for coin in coins)`. The base cases are when s = 0 and s < 0.
```py
def coinChange(self, coins: List[int], amount: int) -> int:
    if len(coins) == 0 or amount <= 0:
          return 0
      result =  self.helper(amount, coins, {})

      return result if result != math.inf else -1

def helper(self, s, coins, memo):
  # helper(s) = helper(s - coin) + 1
  if s == 0:
      return 0
  if s < 0:
      return math.inf
  if s in memo:
      return memo[s]
  
  memo[s] = min(self.helper(s - coin, coins, memo) + 1 for coin in coins)
      
  return memo[s]
```
**7.2 Bottom-top**
```py
def coinChange(coins, amount):
    if len(coins) == 0:
        return 0
    
    if amount <= 0 :
        return 0
    
    dp = [math.inf for _ in range(amount + 1)]
    dp[0] = 0
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i-coin] + 1)
    
    return dp[amount] if dp[amount] != math.inf else -1
```

## 8. Min Falling Path Sum
- https://leetcode.com/problems/minimum-falling-path-sum/
  
**top-bottom**
```py
class Solution:
    def minFallingPathSum(self, matrix: List[List[int]]) -> int:
        if matrix is None:
            return None
        
        if len(matrix) == 1:
            return min(matrix[0])

        if len(matrix[0]) == 1:
            return sum(matrix)

        minVal = math.inf
        memo = {}
        for j in range(len(matrix[0])):
            minVal = min(minVal, self.helper(matrix, 0, j, memo))

        return minVal if minVal != math.inf else -1

    def helper(self, matrix, i, j, memo):
        # dp[i,j] = min(dp[i+1,j], dp[i+1, j-1], dp[i+1, j+1]) + matrix[i][j]
        # base cases
        if j < 0 or j >= len(matrix[0]):
            return math.inf
        if i == len(matrix) - 1:
            return matrix[i][j]
            
        if (i, j) in memo:
            return memo[(i, j)]

        left = self.helper(matrix, i+1, j-1, memo)
        center = self.helper(matrix, i+1, j, memo)
        right = self.helper(matrix, i+1, j+1, memo)

        memo[(i,j)] = min(left, center, right) + matrix[i][j]
        
        return memo[(i,j)]
```
- **Time complexity: O(N^2)**: For every cell in the matrix, we will compute the result only once and update the memo. For the subsequent calls, we are using the stored results that take O(1) time. There are N^2 cells in the matrix, and thus N^2 dp states. So, the time complexity is O(N^2).
- **Space complexityL O(N^2)**: The recursive call stack uses O(N) space. As the maximum depth of the tree is N, we can’t have more than N recursive calls on the call stack at any time. The 2D matrix memo uses O(N^2)space. Thus, the space complexity is `O(N) + O(N^2) = O(N^2)`.
