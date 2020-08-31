"""
Say you have an array for which the ith element is the price of a given stock on day i.

If you were only permitted to complete at most one transaction (i.e., buy one and sell one share of the stock), design an algorithm to find the maximum profit.

Note that you cannot sell a stock before you buy one.

Example 1:

Input: [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
             Not 7-1 = 6, as selling price needs to be larger than buying price.
Example 2:

Input: [7,6,4,3,1]
Output: 0
Explanation: In this case, no transaction is done, i.e. max profit = 0.
"""
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        n = len(prices)
        if n<= 1:
            return 0
        else:

            # Construct array of profit
            previous = prices[0]

            for k in range(n):
                if k==0:
                    prices[k]=0
                else:
                    previous_ = prices[k]
                    prices[k] = prices[k]-previous
                    previous = previous_

            # Keep track of max subarray
            max_sofar = prices[0]
            max_profit = prices[0]


            for item in prices:
                max_sofar = max(0, item+max_sofar)
                max_profit = max(max_sofar, max_profit)

            return max_profit

s = Solution()
s.maxProfit([7,1,5,3,6,4])
