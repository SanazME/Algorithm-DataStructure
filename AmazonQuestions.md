
## 1. Merge k sorted Lists: https://leetcode.com/problems/merge-k-sorted-lists/

- always think a Brute-force approach first!
- Time complexity : `O(NlogN)` where N is the total number of nodes.
  - Collecting all the values costs O(N) time.
  - A stable sorting algorithm costs O(NlogN) time.
  - Iterating for creating the linked list costs O(N) time.
- Space complexity : O(N).

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

- The other approach is to compare every k nodes (head of every linked list) and get the node with the smallest value. Extend the final sorted linked list with the selected nodes. We can use PriorityQueue to save the first elements of all lists in a PriorityQueue and then retrieve the smallest value in the queue first and increment the relevant list node till we finish all of those lists.

- `from Queue import PriorityQueue`, the `PriorityQueue` : The lowest valued entries are retrieved first. A typical pattern for entries is a tuple in the form: `(priority_number, data)`.
```py
from Queue import PriorityQueue

q = PriorityQueue()

q.put((3, 'Read')
q.put((5, "Write'))

OR 

q.put(4)
############

q.empty() # check if it's empty

############

val = q.get()
val, nn = q.get()


```




