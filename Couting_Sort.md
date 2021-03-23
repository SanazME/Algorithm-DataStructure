https://leetcode.com/explore/learn/card/fun-with-arrays/523/conclusion/3228/discuss/347368/Easy-Python-O(n)-Let's-step-through-the-algorithm

This problem can be done in O(n) time by using either elements of counting sort, or complete counting sort (full code at bottom).

For learning purposes, let's do a complete counting sort and then compare against the original array.

This is also going to be stable counting sort, so it could be used to sort a linked list or pairs such as [(GOOG, 1), (MSFT, 1), (FB, 3)].

In this problem k is defined to be small:
1 <= heights.length <= 100
1 <= heights[i] <= 100

Counting sort is linear when k = O(n)

We need to create a frequencies array to count the frequencies of the individual numbers from 0 -> k where k is the max_value +1.

```py
max_value = max(heights)
frequencies = [0] * (max_nr + 1)
Then get frequencies for each number:

for number in heights:
	frequencies[number] += 1
Example 1 [1,1,4,2,1,3] -> frequencies: [0, 3, 1, 1, 1]
```

Next we have to create a sumcount array by adding element + element prior starting at index 1.

```py
sumcount = [0] * (max_nr + 1)
for index, number in enumerate(count[1:],start=1):
	sumcount[index] = number + sumcount[index-1]
Example 1 [1,1,4,2,1,3] -> frequencies: [0, 3, 1, 1, 1] -> sumcount: [0, 3, 4, 5, 6]
```

This is the only tricky part:

The sumcount array determines the index of the number to-be-sorted in the final output array because each value -1 represents the last position, that a number to-be-sorted can occur.

In other words: sumcount[i] now contains the number of elements ≤ i. For example, let's see where 3 from the original input array needs to go:

```
sumcount: [0, 3, 4, 5, 6]
           0  1  2  3  4
```
By looking at index 3, we get value 5. Value 5-1 is 4. From looking at the input array [1,1,4,2,1,3] we can see, yes, 3 needs to be at index 4. The -1 stems from rebasing to 0.

Last, we need to actually place the elements in an output array. For each element, we also decrement the value in the sumcount array so that if the number occurs again, it is placed at the next smaller index. Note that we loop backwards so that the sorting is stable, which is not neccessary when dealing with only integers.

```py
output = [0] * len(heights)
#loop backwards starting with last element
for p in range(len(heights)-1,-1,-1):
    output[sumcount[heights[p]]-1] = heights[p]
    sumcount[heights[p]] -= 1
```
We are done with counting sort and could return the output array. For this problem, we need to compare the final output with the initial input array and return the difference.

```py
result = 0
for index, number in enumerate(heights):
    if number != output[index]:
        result += 1
return result
````

```py
class Solution:
    def heightChecker(self, heights: List[int]) -> int:
        max_nr = max(heights)
        # initialize frequency array with 0's
        count = [0] * (max_nr + 1)
        # get frequencies
        for number in heights:
            count[number] += 1
        # create a sumcount array
        sumcount = [0] * (max_nr + 1)
        for index, number in enumerate(count[1:],start=1):
            sumcount[index] = number + sumcount[index-1]
        # sumcount determines the index in sorted array
        # create output array
        output = [0] * len(heights)
        # loop backwards starting with last element for stable sort
        for p in range(len(heights)-1,-1,-1):
            output[sumcount[heights[p]]-1] = heights[p]
            sumcount[heights[p]] -= 1
		# return the difference compared to original array
        result = 0
        for index, number in enumerate(heights):
            if number != output[index]:
                result += 1
        return result
```
