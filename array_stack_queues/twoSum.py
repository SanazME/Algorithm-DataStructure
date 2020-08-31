# https://leetcode.com/problems/two-sum/


class Solution(object):

    def twoSum(self, nums, target):
        """
        :type nums : List[int]
        :type target : int
        :rtype List [int]
        """
        completeDict = {}
        for i in range(0, len(nums)):
            completeDict[nums[i]] = i
            complementary = target - nums[i]
            if complementary in completeDict.keys():
                return [i, completeDict[complementary]]
        return


s = Solution()
print(s.twoSum([2, 11, 9, 15], 11))
