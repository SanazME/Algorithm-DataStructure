"""
Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

push(x) -- Push element x onto stack.
pop() -- Removes the element on top of the stack.
top() -- Get the top element.
getMin() -- Retrieve the minimum element in the stack.
Example:
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin();   --> Returns -3.
minStack.pop();
minStack.top();      --> Returns 0.
minStack.getMin();   --> Returns -2.


youtube: https://www.youtube.com/watch?v=nGwn8_-6e7w

"""
class MinStack(object):
    def __init__(self):
        """
        initialize your data structure here.
        """
        self._items = []
        

    def push(self, x):
        """
        :type x: int
        :rtype: None
        """
        current_min = self.getMin()
        print('current_min:', current_min)
        if current_min is None or x < current_min:
            current_min = x
        self._items.append((x,current_min))       
        

    def pop(self):
        """
        :rtype: None
        """
        self._items.pop()
        

    def top(self):
        """
        :rtype: int
        """
        if len(self._items)==0:
            return None
        else:
            return self._items[-1][0]
        

    def getMin(self):
        """
        :rtype: int
        """
        if len(self._items)==0:
            return None
        else:
            return self._items[-1][1]

        
obj=MinStack()
obj.push(-10)
obj.push(14)
print(obj.getMin())
print(obj.getMin())
obj.push(-20)
print(obj.getMin())
print(obj.getMin())
obj.top()
print(obj.getMin())
obj.pop()
obj.push(10)
obj.push(-7)
print(obj.getMin())
obj.push(-7)
obj.pop()
obj.top()
print(obj.getMin())
obj.pop()