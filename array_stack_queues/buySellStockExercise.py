class Solution(object):
    def maxProfit(self, price):
        if price is None:
            return 
        else:
            profit = 0
            for i in range(1, len(price)):
                diff = price[i] - price[i-1]
                if diff > 0:
                    profit += diff
            return profit
            
test = Solution()
print(test.maxProfit([7,1,5,3,6,4]))
print(test.maxProfit([1,2,3,4,5]))
print(test.maxProfit([7,6,4,3,1]))
