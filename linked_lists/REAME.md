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
