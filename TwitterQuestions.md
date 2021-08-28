## Write a collection of data structure that is fast (should run O(1) time amortized) for each of the following operation - add(element), remove(ele), get(ele).
- Insert Delete GetRandom O(1): https://leetcode.com/problems/insert-delete-getrandom-o1/
- If we have to implement LRU cache approach, we would use a hash table and DB linked list. The hash table keeps the key and linked list nodes so we can find a node in O(1) time. We can insert the node always after header node so it'll be O(1) time. For removing a node, we find the node in O(1) from the hash table and then we can just remove it in O(1). However, to implement LRU cache, we want to always move the node after the header if we add it or if we make a get query and we want remove the leaset used node if we reach the capacity by moving the last node before the tail node.

- IF we don't implement LRU cache and we just O(1) in all those operations, we can either use a hash table or an **undorted array**: https://en.wikipedia.org/wiki/Search_data_structure.  Insert on an unsorted array is sometimes quoted as being O(n) due to the assumption that the element to be inserted must be inserted at one particular location of the array, which would require shifting all the subsequent elements by one position. However, in a classic array, the array is used to store arbitrary unsorted elements, and hence the exact position of any given element is of no consequence, and insert is carried out by increasing the array size by 1 and storing the element at the end of the array, which is a O(1) operation.Likewise, the deletion operation is sometimes quoted as being O(n) due to the assumption that subsequent elements must be shifted, but in a classic unsorted array the order is unimportant (though elements are implicitly ordered by insert-time), so deletion can be carried out by swapping the element to be deleted with the last element in the array and then decrementing the array size by 1, which is a O(1) operation.

- Now with implementing a getRandom function, we can't only use hashtable or an array. Hashmap provides Insert and Delete in average constant time, although has problems with GetRandom. The idea of GetRandom is to choose a random index and then to retrieve an element with that index. There is no indexes in hashmap, and hence to get true random value, one has first to convert hashmap keys in a list, that would take linear time. The solution here is to build a list of keys aside and to use this list to compute GetRandom in constant time. Array has the issues we mentioned earlier.

- The solution is have a hashtable and an array. Hashtable to store the elements as key and their index as value so we know where they are in the array. To delete a value at arbitrary index takes linear time. The solution here is to always delete the last value:

    - Swap the element to delete with the last one.

    - Pop the last element out.

```py
class RandomizedSet(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.hashTable = {}
        self.arr = []
        self.idx = 0
                
        
    def insert(self, val):
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        :type val: int
        :rtype: bool
        """
        if val in self.hashTable:
            return False
        
        # insert in array
        self.arr.append(val)
                
        # insert in hash table
        self.hashTable[val] = self.idx
        
        # increment index 
        self.idx += 1
        
        return True
        
    def remove(self, val):
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        :type val: int
        :rtype: bool
        """
        if val not in self.hashTable:
            return False
        
        loc = self.hashTable[val]
        
        # if the val is NOT at the end of array
        if loc != self.idx:
            self._swapWithLast(loc)

            
        self.arr.pop()
        self.idx -= 1
        del self.hashTable[val]
        

        return True
        
        
    def _swapWithLast(self, loc):
        # swap with the last element
        self.arr[-1], self.arr[loc] = self.arr[loc], self.arr[-1]
        
        # update hash table value (location of last element)
        self.hashTable[self.arr[loc]] = loc

    def getRandom(self):
        """
        Get a random element from the set.
        :rtype: int
        """
        return random.choice(self.arr)
        

# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()

```
