## 875. Koko Eating Bananas
- https://leetcode.com/problems/koko-eating-bananas/description/?envType=company&envId=amazon&favoriteSlug=amazon-thirty-days
- In the problem, Koko is given n piles of bananas, represented by an integer array of length n. She eats bananas at a constant speed, for example, x bananas per hour. The time taken to eat a pile of y bananas is y/x after rounding up to the closest integer. For example, if she eats 3 bananas per hour, it takes her 2 hours to eat a pile of 4 bananas.

The first constraint of the problem is that Koko has to eat all the piles within h hours, where h is no less than the number of piles. We can imagine that with a fast speed, Koko spends 1 hour on each pile, therefore, she can always finish all the piles within h hours. Let's call this kind of speed workable speed. Likewise, let any eating speed at which Koko can't eat all the piles be unworkable speed.

However, we have another constraint that Koko would like to eat as slow as possible, therefore, among all the workable eating speeds, we need to find out the minimum one.

**Approach 1: Brute Force**
- The brute force approach is to try every possible eating speed to find the smallest workable speed. Starting from speed=1 and incrementing it by 1 each time, we will find a speed at which Koko can eat all piles within h hours, that is, the first minimum speed.
**Algorithm**
1. Start at speed=1.
2. Given the current speed, calculate how many hours Koko needs to eat all of the piles.
    - If Koko cannot finish all piles within h hours, increment speed by 1, that is speed=speed+1 and start over step 2.
    - If Koko can finish all piles within h hours, go to step 3.
3. Return the speed as the answer.
```py
class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        #Start at an eating speed of 1.
        speed = 1

        while True:
            # hour_spent stands for the total hour Koko spends with 
            # the given eating speed.
            hour_spent = 0

            # Iterate over the piles and calculate hour_spent.
            # We increase the hour_spent by ceil(pile / speed)
            for pile in piles:
                hour_spent += math.ceil(pile / speed)    

            # Check if Koko can finish all the piles within h hours,
            # If so, return speed. Otherwise, let speed increment by
            # 1 and repeat the previous iteration.                
            if hour_spent <= h:
                return speed
            else:
                speed += 1
```
- Time: `O(n*m)` - Let n be the length of input array piles and m be the upper bound of elements in piles

**Approach 2: Binary Search**
- In the previous approach, we tried every smaller eating speed, before finding the first workable speed. We shall look for a more efficient way to locate the minimum workable eating speed.

Recall how we calculated the total time for Koko to finish eating all the piles in approach 1. We can observe two laws:

1. If Koko can eat all the piles with a speed of n, she can also finish the task with the speed of n+1. With a larger eating speed, Koko will spend less or equal time on every pile. Thus, the overall time is guaranteed to be less than or equal to that of the speed n.
2. If Koko can't finish with a speed of n, then she can't finish with the speed of n−1 either.
With a smaller eating speed, Koko will spend more or equal time on every pile, thus the overall time will be greater than or equal to that of the speed n.

- **If Koko can finish all the piles within h hours, set right equal to middle signifying that all speeds greater than middle are workable but less desirable by Koko. Otherwise, set left equal to middle+1 signifying that all speeds less than or equal to middle are not workable.**

```py
class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:  
        # Initalize the left and right boundaries     
        left = 1
        right = max(piles)
        
        while left < right:
            # Get the middle index between left and right boundary indexes.
            # hour_spent stands for the total hour Koko spends.
            middle = (left + right) // 2            
            hour_spent = 0
            
            # Iterate over the piles and calculate hour_spent.
            # We increase the hour_spent by ceil(pile / middle)
            for pile in piles:
                hour_spent += math.ceil(pile / middle)
            
            # Check if middle is a workable speed, and cut the search space by half.
            if hour_spent <= h:
                right = middle
            else:
                left = middle + 1
        
        # Once the left and right boundaries coincide, we find the target value,
        # that is, the minimum workable eating speed.
        return right
```
- Time complexity: `O(n⋅logm)`: The initial search space is from 1 to m, it takes logm comparisons to reduce the search space to 1.

## https://github.com/SanazME/Algorithm-DataStructure/blob/master/array_stack_queues/README.md#longest-substring-without-repeating-characters

## 621. Task Scheduler
- https://leetcode.com/problems/task-scheduler/description/?envType=company&envId=amazon&favoriteSlug=amazon-thirty-days
- Ans: https://leetcode.com/problems/task-scheduler/editorial/?envType=company&envId=amazon&favoriteSlug=amazon-thirty-days
```py
from heapq import *
from collections import Counter

class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        if n == 0 or len(tasks) <= 1:
            return len(tasks)

        freq = Counter(tasks)
        queue = [(-v, k) for k, v in freq.items()]
        heapq.heapify(queue)
        time = 0
        arr = []

        print(queue)

        while queue:
            cycle = n + 1
            store = []
            task_count = 0

            while cycle > 0 and queue:
                currFreq, task = heappop(queue)
                print(currFreq, task)
                task_count += 1
                arr.append(task)

                if currFreq * -1 > 1:
                    store.append((currFreq + 1, task))
                cycle -= 1
            # extra
            if len(store) > 0:
                if task_count < n + 1:
                    diff = n + 1 - task_count
                    arr.extend('idle' for _ in range(diff))
            
            for ele in store:
                heapq.heappush(queue, ele)

            if queue:
                time += n + 1
            else:
                time += task_count

        print(arr)
        return time
```
Let the number of tasks be N. Let k be the size of the priority queue. k can, at maximum, be 26 because the priority queue stores the frequency of each distinct task, which is represented by the letters A to Z.

Time complexity: O(N)

In the worst case, all tasks must be processed, and each task might be inserted and extracted from the priority queue. The priority queue operations (insertion and extraction) have a time complexity of O(logk) each. Therefore, the overall time complexity is O(N⋅logk). Since k is at maximum 26, logk is a constant term. We can simplify the time complexity to O(N). This is a linear time complexity with a high constant factor.

Space complexity: O(26) = O(1)

The space complexity is mainly determined by the frequency array and the priority queue. The frequency array has a constant size of 26, and the priority queue can have a maximum size of 26 when all distinct tasks are present. Therefore, the overall space complexity is O(1) or O(26), which is considered constant.

### More similar problems:
767. Reorganize String
1054. Distant Barcodes
1405. Longest Happy String
1953. Maximum Number of Weeks for Which You Can Work
2335. Minimum Amount of Time to Fill Cups
358. Rearrange String k Distance Apart (premium)
984. String Without AAA or BBB

For 
### 1953. Maximum Number of Weeks for Which You Can Work
- we can't use heap because with heap we hit timeline exceeded as the max milestone can be super large. This solution:
```pyclass Solution:
    def numberOfWeeks(self, milestones: List[int]) -> int:
        if len(milestones) == 0:
            return 0

        if len(milestones) == 1:
            return milestones[0]

        queue = [(-v, k) for k, v in enumerate(milestones)]
        heapq.heapify(queue)
        weeks = 0
        prev = None

        while queue:
            cycle = 2
            store = []
            while cycle > 0 and queue:
                currFreq, val = heapq.heappop(queue)
                if val != prev:
                    weeks += 1
                prev = val
                if currFreq * -1 > 1:
                    store.append((currFreq + 1, val))

                cycle -= 1

            for ele in store:
                heapq.heappush(queue, ele)
        
        return weeks
```
- Your solution was simulating the process week by week using a heap
- For large numbers like 355359359, this means millions of heap operations
- The heap-based solution has a time complexity of O(M log N) where M is the maximum milestone value
The intuition behind this solution is:
1. If the maximum project's milestones can be interleaved with other projects `(max_milestone ≤ remaining_sum + 1)`, we can complete everything.
2. Otherwise, we can only use up all the remaining projects' milestones by alternating them with the maximum project, plus one more milestone from the maximum project. For example for [5,2,1] it will be 3*2 + 1 = 7

```py
def numberOfWeeks(self, milestones: List[int]) -> int:
        if len(milestones) == 0:
            return 0

        if len(milestones) == 1:
            return 1

        maxMilestone = max(milestones)
        totalSum = sum(milestones)
        remainingSums = totalSum - maxMilestone

        if maxMilestone <= remainingSums + 1:
            return totalSum
        
        return 2 * remainingSums + 1
```

### 358. Rearrange String k Distance Apart (premium)
Better Cooldown Management:
- Instead of trying to process k+1 characters at once, we process one character at a time
- We keep track of when each character can be used again using a waitQueue
- The waitQueue stores tuples of (next_valid_position, count, char)

```py
def rearrangeString(self, s: str, k: int) -> str:
    if k == 0:
        return s
    if len(s) <= 1:
        return s

    freq = Counter(s)

    queue = [(-v, k) for k, v in freq.items()]
    heapq.heapify(queue)
    result = []
    waitQueue = [] #(next_valid_position, freq, val)

    while queue or waitQueue:
        # check if there are characters in the waitQueue that can be back to queue
        if waitQueue and waitQueue[0][0] <= len(result):
            _, ff, vv = heapq.heappop(waitQueue)
            heapq.heappush(queue, (ff, vv))

        if not queue:
            return ""
       
        currFreq, val = heapq.heappop(queue)
        result.append(val)

        if currFreq * -1 > 1:
            # Add to waitQueue
            heapq.heappush(waitQueue, (len(result) + k - 1, currFreq + 1, val))

            

    return "".join(result)
```

### 2534. Time Taken to Cross the Door
- https://leetcode.com/problems/time-taken-to-cross-the-door/description


  - we keep two queues one for exit and one for enter, we add all entries from arrival to those queues up to where the arrival time stays less or equal to the current time. Then we process stuff in the queues based on the rules. Then we move to the next iteration to add the next batch of arrivals to those queues and increase the curr time by one. Now, if no one is waiting, we fast forward to the next arrival time for curr_time.

```py
ef timeTaken(self, arrival: List[int], state: List[int]) -> List[int]:
        if len(arrival) == 1:
            return arrival

        n = len(arrival)
        curr_time = arrival[0]
        i = 0
        ans = [curr_time for _ in range(len(arrival))]
        enter_q = deque() # Queue for people wanting to enter
        exit_q = deque()  # Queue for people wanting to exit
        prev_state = None  # exit: True, enter: False, None: Not used


        while i < n or exit_q or enter_q:
            # Add new arrivals to respective queues
            while i < n and arrival[i] <= curr_time:
                if state[i] == 0:
                    enter_q.append(i)
                else:
                    exit_q.append(i)

                i += 1
            # Process queues based on rules
            if enter_q or exit_q:
                # Determin who goes next
                use_exit = (
                    exit_q and (
                        not prev_state or # door wasn't used
                        prev_state or # last was exit
                        not enter_q # No one wants to enter
                        )
                )

                if use_exit:
                    person = exit_q.popleft()
                    prev_state = True
                else:
                    person = enter_q.popleft()
                    prev_state = False

                ans[person] = curr_time
                curr_time += 1
            else:
                # No one waiting, fast forward to next arrival
                curr_time = arrival[i] if i < n else curr_time + 1
                prev_state = None

        return ans
```

### 1676. Lowest Common Ancestor of a Binary Tree IV
- https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree-iv/description/

- starting from root, traverse the left and the right subtrees, checking if one of the nodes exist there. If one of subtress doesn't contain any given node, the LCA can be the node returned from the other subtree. If both subtrees contain nodes, the lCA node is the current node.

**Recursive approach**
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', nodes: 'List[TreeNode]') -> 'TreeNode':
        if len(nodes) == 1:
            return nodes[0]

        nodesSet = set(nodes)
        
        return self.helper(root, nodesSet)
        


    def helper(self, node, nodesSet):

        if node is None:
            return None

        if node in nodesSet:
            return node
        
        left = self.helper(node.left, nodesSet)
        right = self.helper(node.right, nodesSet)

        if left and right:
            return node

        return left or right
```
**Iterative approach**
- User post-order traversal (left -> right -> node to count target nodes in each subtree, post-order because when we visit a node, we've already visited and processed its entire subtree.
- when finding a node whose subtree contains all targets, checks if it's the lowest such node by verifying neither child subtrees contains all targets. 

```py
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', nodes: 'List[TreeNode]') -> 'TreeNode':
        if len(nodes) == 1:
            return nodes[0]
            
        nodesSet = set(nodes)
        
        # Store node -> count of target nodes in subtree
        count_map = {}
        stack = [(root, False)]  # (node, visited)
        
        while stack:
            node, visited = stack.pop()
            
            if not node:
                continue
                
            if visited:
                # Post-order visit: calculate count
                count = 0
                if node in nodesSet:
                    count += 1
                if node.left:
                    count += count_map.get(node.left, 0)
                if node.right:
                    count += count_map.get(node.right, 0)
                count_map[node] = count
                
                # If this node's subtree contains all target nodes, check if it's the LCA
                if count == len(nodesSet):
                    left_count = count_map.get(node.left, 0)
                    right_count = count_map.get(node.right, 0)
                    if (left_count != len(nodesSet) and 
                        right_count != len(nodesSet)):
                        return node
            else:
                # Pre-order visit: add children to stack
                stack.append((node, True))
                stack.append((node.right, False))
                stack.append((node.left, False))
                
        return root
```
