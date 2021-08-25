
1. Merge k sorted Lists: https://leetcode.com/problems/merge-k-sorted-lists/

- `arr.sort()` sorts arr in-place. Time complexity is `O(nlog n)`. The function has two optional attributes which can be used to specify a customized sort:
  - `key: sorts the list based on a function or criteria`
  - `reverse: boolean, if true, sort in reverse order`

```py
arr.sort(key=abs, reverse=True)

# A callable function which returns the length of a string
def lengthKey(str):
  return len(str)

list2 = ["London", "Paris", "Copenhagen", "Melbourne"]
# Sort based on the length of the elements
list2.sort(key=lengthKey)
```









