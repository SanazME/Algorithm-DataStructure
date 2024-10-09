## Line Sweep Algorithm
- https://leetcode.com/discuss/study-guide/2166045/line-sweep-algorithms

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
                # Remove entries if their count becomes zero to clean up the SortedDict. So we optimize the space
                if self.booking_count[start] == 0:
                    del self.booking_count[start]

        return True

# Your MyCalendarTwo object will be instantiated and called as such:
# obj = MyCalendarTwo()
# param_1 = obj.book(start,end)
```
## Meeting Rooms II 
- https://github.com/SanazME/Algorithm-DataStructure/blob/master/AmazonQuestions.md#41-meeting-rooms-ii
- Another way of solution using Line Sweep:
```py
def minMeetingRooms(self, intervals: List[List[int]]) -> int:
    map_rooms = SortedDict()

    for interval in intervals:
        start, end = interval[0], interval[1]
        map_rooms[start] = map_rooms.get(start, 0) + 1
        map_rooms[end] = map_rooms.get(end, 0) - 1

    globalMax = 0
    currMax = 0
    for key in map_rooms.keys():
        currMax += map_rooms[key]
        globalMax = max(globalMax, currMax)

    return globalMax
```

## Count positions on street with required brightness
- https://leetcode.com/problems/count-positions-on-street-with-required-brightness/description/

- we don't have to set values for all entries, we just set for start and end + 1 (which there is no light). Then we can compute the factor as we check against requirement slice.
```py
def meetRequirement(self, n: int, lights: List[List[int]], requirement: List[int]) -> int:
    if len(lights) == 0:
        return 0

    map_position = defaultdict(int)

    for light in lights:
        position, rad = light[0], light[1]
        start = max(0, position - rad)
        end = min(n - 1, position + rad) + 1
        map_position[start] += 1
        map_position[end] -= 1

            
    count = 1 if map_position[0] >= requirement[0] else 0
    for i in range(1, len(requirement)):
        map_position[i] += map_position[i - 1]
        if map_position[i] >= requirement[i]:
            count += 1

    return count
```

## Range Addition
- https://leetcode.com/problems/range-addition/description/
```py
def getModifiedArray(self, length: int, updates: List[List[int]]) -> List[int]:
        arr = [0 for _ in range(length)]

        if len(updates) == 0:
            return arr

        for update in updates:
            start, end, inc = update[0], update[1], update[2]
            arr[start] += inc
            if end + 1 <= length - 1:
                arr[end + 1] -= inc

        for i in range(1, length):
            arr[i] += arr[i - 1]

        return arr
```

## Car Pooling
- https://leetcode.com/problems/car-pooling/description/
```py
def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
        trip_map = SortedDict()

        for trip in trips:
            count, start, end = trip[0], trip[1], trip[2]
            trip_map[start] = trip_map.get(start, 0) + count
            trip_map[end] = trip_map.get(end, 0) - count

        globalMax = 0
        currMax = 0

        for key in trip_map.keys():
            currMax += trip_map[key]
            globalMax = max(currMax, globalMax)
            if globalMax > capacity:
                return False

        return True
```

## Burst Balloons
- https://github.com/SanazME/Algorithm-DataStructure/blob/cd312b33bc0c5a4231bb856e9b39eff80487beeb/array_stack_queues/README.md#minimum-number-of-arros-to-burst-balloons

## Non-overlapping Intervals
- https://leetcode.com/problems/non-overlapping-intervals/description/

Finding the minimum number of intervals to remove is equivalent to finding the maximum number of non-overlapping intervals. This is the famous interval scheduling problem.

Let's start by considering the intervals according to their end times. Consider the two intervals with the earliest end times. Let's say the earlier end time is x and the later one is y. We have x < y.

If we can only choose to keep one interval, should we choose the one ending at x or ending at y? To avoid overlap, We should always greedily choose the interval with an earlier end time x. The intuition behind this can be summarized as follows:

- We choose either x or y. Let's call our choice k.
- To avoid overlap, the next interval we choose must have a start time greater than or equal to k.
- We want to maximize the intervals we take (without overlap), so we want to maximize our choices for the next interval.
- Because the next interval must have a start time greater than or equal to k, a larger value of k can never give us more choices than a smaller value of k.
- As such, we should try to minimize k. Therefore, we should always greedily choose x, since x < y.

**Sorting by end time is a greedy algorithm**: This means that it makes the best possible choice at each step, without considering the future. As a result, it is usually more efficient than sorting by start time.

**Sorting by start time is a dynamic programming algorithm**: This means that it makes a choice at each step, based on the choices that it has made in the past. As a result, it is usually more robust to errors in the input data.

```py
def eraseOverlapIntervals(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: int
        """
        if len(intervals) <= 1:
            return 0
        # intervals.sort(key=lambda x:x[0])
        
        intervals.sort(key=lambda x:x[1])

        globalStart, globalEnd = intervals[0][0], intervals[0][1]

        count = 0
        for i in range(1, len(intervals)):
            localStart, localEnd = intervals[i][0], intervals[i][1]
            # globalEnd = min(localEnd, globalEnd)

            if localStart < globalEnd:
                count += 1
            else:
                globalEnd = localEnd

        return count
```

## Insert Interval
- https://leetcode.com/problems/insert-interval/description/
**Solution 1: Using interval**
```py
def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        if len(newInterval) == 0:
            return intervals

        if len(intervals) == 0:
            return [newInterval]

        result = []
        i = 0
        n = len(intervals)

        while i < n and intervals[i][1] < newInterval[0]:
            result.append(intervals[i])
            i += 1

        while i < n and intervals[i][0] <= newInterval[1]:
            newInterval[0] = min(newInterval[0], intervals[i][0])
            newInterval[1] = max(newInterval[1], intervals[i][1])
            i += 1

        result.append(newInterval)

        while i < n:
            result.append(intervals[i])
            i += 1

        return result
```
**Solution 1: using Map Counter**
