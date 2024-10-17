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
