## String to integer
- https://leetcode.com/problems/string-to-integer-atoi/
```py
def myAtoi(self, s):
      """
      :type s: str
      :rtype: int
      """
      s = s.strip()

      if len(s) <= 0:
          return 0

      ls = list(s)

      sign = -1 if s[0] == '-' else 1
      validSet = set(['+', '-'])
      idx = 0

      if s[0] in validSet:
          idx += 1

      result = 0

      while idx < len(ls) and s[idx].isdigit():
          result = result*10 + ord(s[idx]) - ord('0')
          idx += 1

      if result == 0:
          return result
      elif sign * result > 0:
          return min(result, 2**31 - 1)
      elif sign * result < 0:
          return -min(result, 2**31)
```
- `ord()` method converts a character into its Unicode code value. 
    - Try to avoid using `int` and use `ord` in problems when we convert a string to integer because `int` conversion increases time significantly. To convert a string to a integer:
    ```py
    numStr = '4'
    num = ord(numStr) - ord('0')  => that will give num = 4
    ```
    - to first check if a char is digit or not
    ```py
    s[i].isdigit()
    ```
    - `ord()` is useful, if you want to check whether a string contains a character. `ord(char)`. We can also create an array of all characters with their Unicode value as the key and their frequency in a string as a value:
```py
chars = [0]*128
for char in s:
    chars[ord(char)] += 1
```
- To remove leading and trailing whitespaces:
    - `s.strip()` : returns **a new string** after removing **any leading and trailing whitespaces including tabs (\t)**
    - `s.lstrip()` : returns **a new string** after removing **any leading and trailing whitespaces from the left of the string**
    - `s.rstrip()` : returns **a new string** after removing **any leading and trailing whitespaces from right of the string **


- https://leetcode.com/problems/group-anagrams/
**solution**
Maintain a map ans : {String -> List} where each key `K` is a sorted string, and each value is the list of strings from the initial input that when sorted, are equal to `K`.

In Java, we will store the key as a string, eg. code. In Python, we will store the key as a hashable tuple, eg. `('c', 'o', 'd', 'e')`.

```py
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        if len(strs) <= 1:
            return [strs]
        
        dic = {}
        
        for st in strs:
            key = tuple(sorted(st))
            
            if key in dic:
                dic[key].append(st)
            else:
                dic[key] = [st]
                
        re = []
        
        for k in dic.keys():
            re.append(dic[k])
            
        return re
```
