# https://leetcode.com/problems/contains-duplicate/


class Solution(object):

    def containsDuplicate(self, nums):
        """
        :type nums : List[int]
        :rtype : bool
        """
        numSet = set()
        for number in nums:
            if number in numSet:
                retur True
            else:
                numSet.add(number)
        return False


S = Solution()
print(S.containsDuplicate([1, 2, 3, 1]))
print(S.containsDuplicate([1, 2, 3, 5]))
print(S.containsDuplicate([1, 1, 1, 3, 3, 4, 3, 2, 4, 2]))
