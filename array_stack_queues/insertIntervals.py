"""
coderbyte
Insert an interval into a list of sorted disjoint intervals
This is a common interview question where the input is a sorted list of disjoint intervals, and your goal is to insert a new interval and merge all necessary intervals returning a final new list. For example, if the interval list is [[1,5], [10,15], [20,25]] and you need to insert the interval [12,27], then your program should return the new list: [[1,5], [10,27]].

Algorithm
  
(1) Create an array where the final intervals will be stored.
(2) Push all the intervals into this array that come before the new interval you are adding.
(3) Once we reach an interval in that comes after the new interval, add our new interval 
to the final array.
(4) From this point, check each remaining element in the array and determine if the intervals 
need to be merged.

"""
def insertIntervals(arr, interval):
    if len(arr)==0:
        return
    
    result = []
    endset=[]
    i=0
    
    while i<len(arr) and arr[i][1]<interval[0]:
        result.append(arr[i])
        i+=1
        
    result.append(interval)
    
    while i<len(arr):
        last = result[-1]
        if arr[i][0] < last[1]:
            minElem = min(arr[i][0], last[0])
            maxElem = max(arr[i][1], last[1])
            result[-1] = [minElem, maxElem]
        else:
            endset.append(arr[i])        
        i+=1
    return result+endset

print(insertIntervals([[1,5], [10,15], [20,25], [30,90]], [12,27]))
