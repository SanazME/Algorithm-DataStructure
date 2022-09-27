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

- https://leetcode.com/problems/implement-strstr/
- We iterate through the haystack and needle:
1. if we're towards the end of haystack that the we can't iterate through the lenght of needle, go to the next index of haystack
2. within inner loop, iterate through needle with fix indx in haystack and compare element by element, if one element was not the same break from inner loop
3. before exisitng inner loop, check if we reach the end of needle, meaning we found the match, return the index of haystack


```py
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        if len(needle) == 0:
            return 0
        
        pointer = 0
        
        for i in range(len(haystack)):
            
            if i + len(needle) > len(haystack):
                break
            
            for j in range(len(needle)):
                if needle[j] != haystack[i + j]:
                    break
                    
                if j == len(needle) - 1:
                    return i
                
        return -1
```
## Add Binary
- https://leetcode.com/problems/add-binary/

```py
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        size_a = len(a)
        size_b = len(b)

        if size_a == 0:
            return b
        if size_b == 0:
            return a

        maxSize = max(size_a, size_b)
        a = a.zfill(maxSize)
        b = b.zfill(maxSize)
        output = ''
        carry = 0
        idx = 0

        for i in range(maxSize - 1, -1, -1):
            
            if a[i] == '1' and b[i] == '1':
                if carry == 0:
                    output = '0' + output
                    carry = 1
                else:
                    output = '1' + output
                    carry = 1
            elif a[i] == '1' or b[i] == '1':
                if carry == 1:
                    output = '0' + output
                else:
                    output = '1' + output     
            else:
                if carry == 1:
                    output = '1' + output
                    carry = 0
                else:
                    output = '0' + output
        
        if carry == 1:
            output = '1' + output

        return output
```
OR 
```py
class Solution:
    def addBinary(self, a, b) -> str:
        n = max(len(a), len(b))
        a, b = a.zfill(n), b.zfill(n)
        
        carry = 0
        answer = []
        for i in range(n - 1, -1, -1):
            if a[i] == '1':
                carry += 1
            if b[i] == '1':
                carry += 1
                
            if carry % 2 == 1:
                answer.append('1')
            else:
                answer.append('0')
            
            carry //= 2
        
        if carry == 1:
            answer.append('1')
        answer.reverse()
        
        return ''.join(answer)
```
