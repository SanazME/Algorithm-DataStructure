"""
Given an array of n positive integers and a positive integer s, find the minimal length of a contiguous subarray of which the sum ≥ s. If there isn't one, return 0 instead.

Example: 

Input: s = 7, nums = [2,3,1,2,4,3]
Output: 2
Explanation: the subarray [4,3] has the minimal length under the problem constraint.

https://leetcode.com/problems/minimum-size-subarray-sum/solution/


https://medium.com/@ratulsaha/preparing-for-programming-interview-as-a-phd-student-with-python-5f8af8b40d5f

"""

class Solution(object):
    def minSubArrayLen(self, s, nums):
        """
        :type s: int
        :type nums: List[int]
        :rtype: int
        """
        sum_local = 0
        pointer = 0
        count = float('Inf')
        
        
        for i in range(len(nums)):
            sum_local += nums[i]
            
            while (sum_local >= s):
                print(i)
                print(sum_local)
                count = min(i - pointer + 1, count)
                print('counter :', count)
                sum_local -= nums[pointer]
                pointer += 1
                print('sum_local :', sum_local)
                print('********')
                
        if count != float('Inf'):
            return count
        else:
            return 0
                
                
# Brute force with memoization of sums
"""
Approach #2 A better brute force [Accepted]
Intuition

In Approach #1, you may notice that the sum is calculated for every surarray in O(n)O(n) time. But, we could easily find the sum in O(1) time by storing the cumulative sum from the beginning(Memoization). After we have stored the cumulative sum in \text{sums}sums, we could easily find the sum of any subarray from ii to jj.

Algorithm

The algorithm is similar to Approach #1.
The only difference is in the way of finding the sum of subarrays:
Create a vector \text{sums}sums of size of \text{nums}nums
Initialize \text{sums}[0]=\text{nums}[0]sums[0]=nums[0]
Iterate over the \text{sums}sums vector:
Update \text{sums}[i] = \text{sums}[i-1] + \text{nums}[i]sums[i]=sums[i−1]+nums[i]
Sum of subarray from ii to jj is calculated as: \text{sum}=\text{sums}[j] - \text{sums}[i] +\text{nums}[i]sum=sums[j]−sums[i]+nums[i], , wherein \text{sums}[j] - \text{sums}[i]sums[j]−sums[i] is the sum from (i+1i+1)th element to the jjth element.


Complexity analysis

Time complexity: O(n^2)O(n 
2
 ).

Time complexity to find all the subarrays is O(n^2)O(n 
2
 ).
Sum of the subarrays is calculated in O(1)O(1) time.
Thus, the total time complexity: O(n^2 * 1) = O(n^2)O(n 
2
 ∗1)=O(n 
2
 )
Space complexity: O(n)O(n) extra space.

Additional O(n)O(n) space for \text{sums}sums vector than in Approach #1.
"""
class Solution2(object):
    def minSubArrayLen(self, s, nums):
        """
        :type s: int
        :type nums: List[int]
        :rtype: int
        """
        count = float('Inf')
        
        sum_vec = [0]*len(nums)
        sum_vec[0] = nums[0]
        
        for i in range(1, len(nums)):
            sum_vec[i] = sum_vec[i-1] + nums[i]
        
        
        for i in range(len(nums)):
            for j in range(i, len(nums)):
                sum_local = sum_vec[j] - sum_vec[i] + nums[i]
                
                if (sum_local >= s):
                    count = min(count, j-i+1)
                
                    break
        
        if count != float('Inf'):
            return count
        else:
            return 0