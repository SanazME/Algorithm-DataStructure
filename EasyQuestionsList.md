https://leetcode.com/list/?selectedList=o160a5j5#

## 1. Min Stack
### Solution 1
- Create an array and each element in the array is a set of `(val, minVal)` and this way we can track the min value in stack for every node.
```py
class MinStack:

    def __init__(self):
        self.stack = []
        

    def push(self, x: int) -> None:
        
        # If the stack is empty, then the min value
        # must just be the first value we add
        if not self.stack:
            self.stack.append((x, x))
            return

        current_min = self.stack[-1][1]
        self.stack.append((x, min(x, current_min)))
        
        
    def pop(self) -> None:
        self.stack.pop()
        

    def top(self) -> int:
        return self.stack[-1][0]
        

    def getMin(self) -> int:
        return self.stack[-1][1]
```
- **Time complexity: O(1) for all operations**
- **Space complexity: O(n)**


### Solution 2
- use a linked list and each node has a min val, we push and pop the head.
```py
class Node:
    def __init__(self, val = None):
        self.val = val
        self.next = None
        self.min = float('Inf')

class MinStack:
    def __init__(self):
        self.head = Node()
        
    def push(self, val):
        curr = self.head
        node = Node(val)
        node.min = min(curr.min, val)
        node.next = curr
        self.head = node
        
    def pop(self):
        curr = self.head
        nextNode = self.head.next
        curr.next = None
        self.head = nextNode
        
        
    def top(self):
        return self.head.val
        
        
    def getMin(self):
        return self.head.min
```

### Solution 3
- TWo stacks: Approach 1 required storing two values in each slot of the underlying Stack. Sometimes though, the minimum values are very repetitive. Do we actually need to store the same minimum value over and over again?
- Instead of only pushing numbers to the min-tracker Stack if they are less than the current minimum, we should push them if they are less than or equal to it. While this means that some duplicates are added to the min-tracker Stack, the bug will no longer occur. Here is another animation with the same test case as above, but the bug fixed.
```py
class MinStack:

    def __init__(self):
        self.stack = []
        self.min_stack = []        
        

    def push(self, x: int) -> None:
        self.stack.append(x)
        if not self.min_stack or x <= self.min_stack[-1]:
            self.min_stack.append(x)
    
    def pop(self) -> None:
        if self.min_stack[-1] == self.stack[-1]:
            self.min_stack.pop()
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]
```

### Solution 3 Improvement
- In the above approach, we pushed a new number onto the min-tracker Stack if, and only if, it was less than or equal to the current minimum.

One downside of this solution is that if the same number is pushed repeatedly onto MinStack, and that number also happens to be the current minimum, there'll be a lot of needless repetition on the min-tracker Stack. Recall that we put this repetition in to prevent a bug from occurring (refer to Approach 2).
An improvement is to put pairs onto the min-tracker Stack. The first value of the pair would be the same as before, and the second value would be how many times that minimum was repeated.

```py
class MinStack:

    def __init__(self):
        self.stack = []
        self.min_stack = []        
        

    def push(self, x: int) -> None:
        
        # We always put the number onto the main stack.
        self.stack.append(x)
        
        # If the min stack is empty, or this number is smaller than
        # the top of the min stack, put it on with a count of 1.
        if not self.min_stack or x < self.min_stack[-1][0]:
            self.min_stack.append([x, 1])
            
        # Else if this number is equal to what's currently at the top
        # of the min stack, then increment the count at the top by 1.
        elif x == self.min_stack[-1][0]:
            self.min_stack[-1][1] += 1

    
    def pop(self) -> None:

        # If the top of min stack is the same as the top of stack
        # then we need to decrement the count at the top by 1.
        if self.min_stack[-1][0] == self.stack[-1]:
            self.min_stack[-1][1] -= 1
            
        # If the count at the top of min stack is now 0, then remove
        # that value as we're done with it.
        if self.min_stack[-1][1] == 0:
            self.min_stack.pop()
            
        # And like before, pop the top of the main stack.
        self.stack.pop()


    def top(self) -> int:
        return self.stack[-1]


    def getMin(self) -> int:
        return self.min_stack[-1][0]   
```

## 2. Intersection of Two Linked Lists
- Time complexity: `O(N + M)` where `N` and `M` are lenght of linked lists
- Space complexity: `O(1)`

```py
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        pa, pb = headA, headB
        lena, lenb = 0, 0

        while pa.next:
            pa = pa.next
            lena += 1
        lena += 1

        while pb.next:
            pb = pb.next
            lenb += 1
        lenb += 1

        if pa.val != pb.val:
            return None

        if lena > lenb:
            pLong, pShort = headA, headB
            countLong, countShort = lena, lenb
        else:
            pLong, pShort = headB, headA
            countLong, countShort = lenb, lena
        
        while countLong > countShort:
            pLong = pLong.next
            countLong -= 1

        # both lists have the same length
        while pLong:
            if pLong == pShort:
                return pLong
            
            pLong = pLong.next
            pShort = pShort.next

        return None
```

## 3. Rotate Array
- https://github.com/SanazME/Algorithm-DataStructure/blob/master/array_stack_queues/README.md#rotate-array

## 4. Happy Number
- https://github.com/SanazME/Algorithm-DataStructure/blob/master/AmazonQuestions3.md#53-happy-number
