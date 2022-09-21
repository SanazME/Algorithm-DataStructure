# Merge Sort
- Divide and conquer
- Time Complexity	
- Best	O(n*log n)
- Worst	O(n*log n)
- Average	O(n*log n)
- Space Complexity	O(n)
- Stability	Yes

```py
def mergeSort(arr):
    if len(arr) > 1:
        
        # mid is the point where the array is divided into two subarrays
        mid = len(arr) // 2
        L = arr[:mid]
        R = arr[mid:]
        
        # Sort the two halves
        mergeSort(L)
        mergeSort(R)
        
        # Merge the two halves
        i = j = k = 0
        
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
            
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
            
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1
```

# Quick Sort
- Divide and conquer
- Time Complexity	
- Best	O(n*log n)
- Worst	O(n^2)
- Average	O(n*log n)
- Space Complexity	O(log n)
- Stability	No


**Worst Case Complexity [Big-O]: O(n2)**
It occurs when the pivot element picked is either the greatest or the smallest element.
This condition leads to the case in which the pivot element lies in an extreme end of the sorted array. 
One sub-array is always empty and another sub-array contains n - 1 elements. Thus, quicksort is called only on this sub-array.

**Best Case Complexity [Big-omega]: O(n*log n)**
It occurs when the pivot element is always the middle element or near to the middle element.

```py
# Quick sort in Python

# function to find the partition position
def partition(array, low, high):

  # choose the rightmost element as pivot
  pivot = array[high]

  # pointer for greater element
  i = low - 1

  # traverse through all elements
  # compare each element with pivot
  for j in range(low, high):
    if array[j] <= pivot:
      # if element smaller than pivot is found
      # swap it with the greater element pointed by i
      i = i + 1

      # swapping element at i with element at j
      (array[i], array[j]) = (array[j], array[i])

  # swap the pivot element with the greater element specified by i
  (array[i + 1], array[high]) = (array[high], array[i + 1])

  # return the position from where partition is done
  return i + 1

# function to perform quicksort
def quickSort(array, low, high):
  if low < high:

    # find pivot element such that
    # element smaller than pivot are on the left
    # element greater than pivot are on the right
    pi = partition(array, low, high)

    # recursive call on the left of pivot
    quickSort(array, low, pi - 1)

    # recursive call on the right of pivot
    quickSort(array, pi + 1, high)


data = [8, 7, 2, 1, 0, 9, 6]
print("Unsorted Array")
print(data)

size = len(data)

quickSort(data, 0, size - 1)

print('Sorted Array in Ascending Order:')
print(data)
```

