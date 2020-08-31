"""
Write a function to reverse a Linked List in place. The function will take in the head of the list as input and return the new head of the list.

You are given the example Linked List Node class:
"""

class Node(object):
    def __init__(self,value):
        self.value=value
        self.next=None

def reverseLinkedList(head):
    prev=None
    current=head

    while current:
        nextNode = current.next
        current.next = prev
        prev, current = current, nextNode

    return prev


node1=Node(1)
node2=Node(2)
node3=Node(3)
node4=Node(4)

node1.next = node2
node2.next = node3
node3.next = node4

newhead = reverseLinkedList(node1)
print(newhead.value)
print(newhead.next.value)