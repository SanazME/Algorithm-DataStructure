"""
https://www.youtube.com/watch?v=OvpKeraoxW0

Question: You are given a standard linked list with next pointer BUT each node carries an additional random pointer to any given node in the linked list. Clone the linked list.

Time O(n), Space O(n) because of the hash table. We can optimize the space to O(1) by just inserting new
copied linked list cells between the current linked list cells and rewire them.

"""

class LinkedList_(object):
    def __init__(self, value, nextNode=None, randomNode=None):
        self.value = value
        self.nextNode = nextNode
        self.randomNode = randomNode

    def printNodes(self):
        node=self

        while node:
            print(node.value, node.nextNode.value if node.nextNode else -1, \
                node.randomNode.value if node.randomNode else -1)

            node=node.nextNode

    def copyList(self):
        node=self
        mapNodes={}
        # head of copied list
        head = LinkedList(node.value)
        mapNodes[node] = head
        
        while node.nextNode:
            mapNodes[node.nextNode] = LinkedList(node.nextNode.value)

            mapNodes[node].nextNode = mapNodes[node.nextNode]

            node = node.nextNode

        node=self
        while node:
            mapNodes[node].randomNode = mapNodes[node.randomNode]

            node=node.nextNode
        
        return head


# Space O(1)
class LinkedList(object):
    def __init__(self, value, nextNode=None, randomNode=None):
        self.value=value
        self.nextNode = nextNode
        self.randomNode = randomNode

    def printNodes(self):
        node=self

        while node:
            print(node.value, node.nextNode.value if node.nextNode else -1, \
                node.randomNode.value if node.randomNode else -1)

            node = node.nextNode

    def copyList(self):
        node=self
        # Add clone nodes after each node + next pointer
        while node:
            newNode = LinkedList(node.value, node.nextNode)
            node.nextNode = newNode
            node = node.nextNode.nextNode

        # Add random pointers
        node=self
        while node:
            node.nextNode.randomNode = node.randomNode.nextNode
            node=node.nextNode.nextNode

        # Split original and new linked lists
        node=self
        newHead = node.nextNode
        newNode = newHead

        while node:
            node.nextNode = node.nextNode.nextNode
            newNode.nextNode = newNode.nextNode.nextNode if newNode.nextNode else None

            node = node.nextNode if node else None
            newNode = newNode.nextNode if newNode else None

        return newHead


if __name__=="__main__":
    node1 = LinkedList("1")
    node2 = LinkedList("2")
    node3 = LinkedList("3")
    node4 = LinkedList("4")
    node5 = LinkedList("5")

    node1.nextNode = node2
    node2.nextNode = node3
    node3.nextNode = node4
    node4.nextNode = node5

    node1.randomNode = node3
    node2.randomNode = node4
    node3.randomNode = node5
    node4.randomNode = node1
    node5.randomNode = node2
    
    newList = node1.copyList()
    newList.printNodes()





