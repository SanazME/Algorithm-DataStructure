## Linked List
- Unlike the array, we are not able to access a random element in a singly-linked list in constant time. 
- If we want to get the ith element, we have to traverse from the head node one by one. It takes us O(N) time on average to visit an element by index, where N is the length of the linked list.
- You might wonder why the linked list is useful though it has such a bad performance (compared to the array) in accessing data by index. We will introduce the **insert and delete operations**.
- **add and insertion**: Unlike an array, we don’t need to move all elements past the inserted element. Therefore, you can insert a new node into a linked list in O(1) time complexity,
- **Delete a node**: 
- If we want to delete an existing node *cur* from the singly linked list, we can do it in two steps:
- 1. Find *cur*'s previous node *prev* and its next node *next*;
- 2. Link *prev* to *cur*'s next node *next*.
In our first step, we need to find out *prev* and *next*. It is easy to find out *next* using the reference field of *cur*. However, we have to traverse the linked list from the head node to find out *prev* which will take O(N) time on average, where N is the length of the linked list. So the time complexity of deleting a node will be O(N).

- Each node in a linked list can be defined by a class with a val and next:
```py
class Node(object):
  def __init__(self, val):
      self.val = val
      self.next = None
```
- We can implement a linked-list defined by the head and the size of the list. We can implement the following methods:
  - `MyLinkedList()` :  Initializes the `MyLinkedList` object.
  - `get(index)` :  Get the value of the indexth node in the linked list. If the index is invalid, return -1.
  - `void addAtHead(int val)` Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list.
  - `void addAtTail(int val)` :  Append a node of value val as the last element of the linked list.
  - `void addAtIndex(int index, int val)`:  Add a node of value val before the indexth node in the linked list. If index equals the length of the linked list, the node will be appended to the end of the linked list. If index is greater than the length, the node will not be inserted.
  - `void deleteAtIndex(int index)` :  Delete the indexth node in the linked list, if the index is valid.
```py
  class Node(object):
    def __init__(self, val):
        self.val = val
        self.next = None

class MyLinkedList(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.head = None
        self.size = 0
        

    def get(self, index):
        """
        Get the value of the index-th node in the linked list. If the index is invalid, return -1.
        :type index: int
        :rtype: int
        """
        if self.size == 0: return -1
        
        curr = self.head
        for i in range(self.size):
            if i == index: return curr.val
            curr = curr.next
        return -1
                

    def addAtHead(self, val):
        """
        Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list.
        :type val: int
        :rtype: None
        """
        node = Node(val)
        node.next = self.head
        self.head = node
        self.size += 1
        

    def addAtTail(self, val):
        """
        Append a node of value val to the last element of the linked list.
        :type val: int
        :rtype: None
        """
        if self.size == 0:
            self.addAtHead(val)
        else:
            node = Node(val)
            curr = self.head
            
            for i in range(1,self.size):
                curr = curr.next
            curr.next = node
            
        self.size += 1

        

    def addAtIndex(self, index, val):
        """
        Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted.
        :type index: int
        :type val: int
        :rtype: None
        """
        if index < 0 or index > self.size: return
        
        if index == self.size:
            self.addAtTail(val)
            return
        if index == 0:
            self.addAtHead(val)
            return
        else:
            node = Node(val)
            curr = self.head
            
            for i in range(1, index):
                curr = curr.next
            node.next = curr.next
            curr.next = node
            self.size += 1
            return
        

    def deleteAtIndex(self, index):
        """
        Delete the index-th node in the linked list, if the index is valid.
        :type index: int
        :rtype: None
        """
        if self.size == 0: return
        
        if index < 0 or index >= self.size: return
        
        if index == 0:
            self.head = self.head.next
            self.size -= 1
            
        else:
            curr = self.head
            prev = None
            for i in range(1, self.size):
                prev = curr
                curr = curr.next
                
                if i == index:
                    prev.next = curr.next
                    self.size -= 1
                    return
        return
             
        # Your MyLinkedList object will be instantiated and called as such:
        # obj = MyLinkedList()
        # param_1 = obj.get(index)
        # obj.addAtHead(val)
        # obj.addAtTail(val)
        # obj.addAtIndex(index,val)
        # obj.deleteAtIndex(index)
 ``` 
 
 - To find the index of the start of a loop in a cyclic linked-list:
    1. find out if the linked-list is cyclic or not and if so, return the node where `slow == fast`
    2. define another function with input: head and the node from previous calculations and return the node where the two nodes reaches eachother (https://www.youtube.com/watch?v=zbozWoMgKW0&t=2s) - based on Floyd's cycle detecting algorithm, the `distance from head to the start of the cycle is always == distance from slow and fast meets to the start of the cycle`

- It is easy to analyze the space complexity. If you only use pointers without any other extra space, the space complexity will be O(1). However, it is more difficult to analyze the time complexity. In order to get the answer, we need to analyze how many times we will run our loop. In our previous finding cycle example, let's assume that we move the faster pointer 2 steps each time and move the slower pointer 1 step each time.
  1. If there is no cycle, the fast pointer takes N/2 times to reach the end of the linked list, where N is the length of the linked list.
  2. If there is a cycle, the fast pointer needs M times to catch up the slower pointer, where M is the length of the cycle in the list.
Obviously, M <= N. So we will run the loop up to N times. And for each loop, we only need constant time. So, the time complexity of this algorithm is O(N) in total.

### Reverse a Linked-List
- One solution is to iterate the nodes in original order and move them to the head of the list one by one. (https://leetcode.com/explore/learn/card/linked-list/219/classic-problems/1204/)
- Therefore, the time complexity is O(N), where N is the length of the linked list. We only use constant extra space so the space complexity is O(1).
- This problem is the foundation of many linked-list problems you might come across in your interview.
- 
**1. Iterative :**
- Assume that we have linked list 1 → 2 → 3 → Ø, we would like to change it to Ø ← 1 ← 2 ← 3.

While you are traversing the list, change the current node's next pointer to point to its previous element. Since a node does not have reference to its previous node, you must store its previous element beforehand. You also need another pointer to store the next node before changing the reference. Do not forget to return the new head reference at the end!
```py
def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        # zero or one-element list
        if head is None or head.next is None: return head

        curr = head
        prev = None

        while curr is not None:
            nextNode = curr.next
            curr.next = prev
            prev = curr
            curr = nextNode
            

        return prev
 ```
**2. Recursive :**
- The recursive version is slightly trickier and the key is to work backwards. Assume that the rest of the list had already been reversed, now how do I reverse the front part? Let's assume the list is: n1 → … → nk-1 → nk → nk+1 → … → nm → Ø

Assume from node nk+1 to nm had been reversed and you are at node nk.

n1 → … → nk-1 → nk → nk+1 ← … ← nm

We want nk+1’s next node to point to nk.

So,

nk.next.next = nk;

Be very careful that n1's next must point to Ø. If you forget about this, your linked list has a cycle in it. This bug could be caught if you test your code with a linked list of size 2.
```py
def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head is None or head.next is None: return head
        p = self.reverseList(head.next)
        head.next.next = head
        head.next = None
        return p
```
