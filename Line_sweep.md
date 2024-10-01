## My Calendar II 
- https://leetcode.com/problems/my-calendar-ii/description/

**Solution 1**
- finding the double booked intevals and use them to check the coming intervals.
```py
class MyCalendarTwo:

    def __init__(self):
        self.bookings = []
        self.double_bookings = []
        

    def book(self, start: int, end: int) -> bool:
        # print(f"start:{start}, end:{end}")
        
        for interval in self.double_bookings:
            start1, end1 = interval[0], interval[1]
            if self._isOverlapping(start1, end1, start, end):
                return False

        for interval in self.bookings:
            start1, end1 = interval[0], interval[1]
            if self._isOverlapping(start1, end1, start, end):
                overlap = self._getOverlap(start1, end1, start, end)
                self.double_bookings.append(overlap)

        self.bookings.append((start, end))
        # print(f"double_booking:{self.double_bookings}")
        return True

    def _isOverlapping(self, start1, end1, start2, end2):
        return max(start1, start2) <= min(end1 - 1, end2 - 1)

    def _getOverlap(self, start1, end1, start2, end2):
        return (max(start1, start2), min(end1, end2))
        


# Your MyCalendarTwo object will be instantiated and called as such:
# obj = MyCalendarTwo()
# param_1 = obj.book(start,end)
```

  **Solution 2**
  - Solution 1 won't be flexible if the requirement was blocking qudraple bookings instead of triple bookings. A more flexible solution is using **Line Sweep** algorithm.
  - The previous approach works well for the given problem, where we need to avoid triple bookings. However, if the requirements change such as checking for four overlapping bookings, the method becomes less flexible. We'd need to introduce additional lists, for example, to track triple bookings, making the solution harder to maintain and extend. To address this, we can use a more flexible and standard solution: the Line Sweep algorithm. This approach is common for interval-related problems and can easily handle changes, such as checking for four or more overlapping bookings.

- The Line Sweep algorithm works by marking when bookings start and end. For each booking `(start, end)`, we mark the start point by increasing its count by 1 (indicating a booking begins), and we mark the end point by decreasing its count by 1 (indicating a booking ends). These marks are stored in a map, which keeps track of the number of bookings starting or ending at each point.

- Once all bookings are processed, we compute the prefix sum over the map. The prefix sum at any point tells us how many active bookings overlap at that moment. If the sum at any point exceeds 2, it means we have a triple booking. At this point, the function should return false to prevent adding a new booking. If no triple booking is found, the function returns true, and the booking is allowed.

```py
from sortedcontainers import SortedDict
class MyCalendarTwo:

    def __init__(self):
        self.overlap_bookings = SortedDict()
        self.maxOverlap = 2

        

    def book(self, start: int, end: int) -> bool:
        self.overlap_bookings[start] = self.overlap_bookings.get(start, 0) + 1
        self.overlap_bookings[end] = self.overlap_bookings.get(end, 0) - 1

        overlapCount = 0
        for key in self.overlap_bookings.keys():
            overlapCount += self.overlap_bookings[key]
            if overlapCount > 2:
                # roll back changes
                self.overlap_bookings[start] -= 1
                self.overlap_bookings[end] += 1
                return False


        return True

        

# Your MyCalendarTwo object will be instantiated and called as such:
# obj = MyCalendarTwo()
# param_1 = obj.book(start,end)
```
