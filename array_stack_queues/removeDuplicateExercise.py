class Solution(object):
    def removeDuplicate(self,nums):
        """
        input : nums [list]
        output: int
        """
        if len(nums)<1:
            return
            
        else:
            pointer = 0
            count = 1

            for i in range(1,len(nums)):
                if nums[i]!= nums[pointer]:
                    pointer += 1
                    count += 1
                    nums[pointer] = nums[i]

        return count

test = Solution()
print(test.removeDuplicate([1,2,2,3,4,5,5,5]))

