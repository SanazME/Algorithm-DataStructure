## 1. Longest Palindromic Substring
- https://leetcode.com/problems/longest-palindromic-substring/
```py
def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """           
        if len(s) <= 1:
            return s
    
        soFar = ''
        for i, char in enumerate(s):
            # for odd case: 'aba'
            tmp = self.helper(s, i, i)
            if len(tmp) >= len(soFar):
                soFar = tmp

            # for even case: 'abba'
            tmp = self.helper(s, i, i+1)

            if len(tmp) >= len(soFar):
                soFar = tmp

        return soFar
        
        
  def helper(self, s, left, right):
      while left >= 0 and right < len(s) and s[left] == s[right]:
          left -= 1
          right += 1
      return s[left+1:right]
```
