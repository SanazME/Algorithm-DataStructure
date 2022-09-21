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

# Counting sort
- Counting sort is a sorting algorithm that sorts the elements of an array by counting the number of occurrences of each unique element in the array. The count is stored in an auxiliary array and the sorting is done by mapping the count as an index of the auxiliary array.
- It is useful when the the range of elements: k = largest number - smallest number is not too large and the number of elements in arr: n is close to k (range of elements) values.

- Time Complexity	
- Best	O(n+k)
- Worst	O(n+k)
- Average	O(n+k)
- Space Complexity	O(k)
- Stability	Yes

```py
# Counting sort in Python programming


def countingSort(array):
    size = len(array)
    output = [0] * size

    # Initialize count array
    count = [0] * 10

    # Store the count of each elements in count array
    for i in range(0, size):
        count[array[i]] += 1

    # Store the cummulative count
    for i in range(1, 10):
        count[i] += count[i - 1]

    # Find the index of each element of the original array in count array
    # place the elements in output array
    i = size - 1
    while i >= 0:
        output[count[array[i]] - 1] = array[i]
        count[array[i]] -= 1
        i -= 1

    # Copy the sorted elements into original array
    for i in range(0, size):
        array[i] = output[i]


data = [4, 2, 2, 8, 3, 3, 1]
countingSort(data)
print("Sorted Array in Ascending Order: ")
print(data)
```

## Heap sort
- Based on heap data structure which a complete binary tree where is either max-heap or min-heap
- In a complete binary tree, the last non leaf node has index of n/2 - 1 (n is number of elements)
- a node with index i has children at indices: 2i+1 and 2i+2
- a node with index i has a parent (lower bound of) n/2 - 1

- Time Complexity	
- Best	O(n*log n)
- Worst	O(n*log n)
- Average	O(n*log n)
- Space Complexity	O(1)
- Stability	No

```py
# Heap Sort in python


  def heapify(arr, n, i):
      # Find largest among root and children
      largest = i
      l = 2 * i + 1
      r = 2 * i + 2
  
      if l < n and arr[i] < arr[l]:
          largest = l
  
      if r < n and arr[largest] < arr[r]:
          largest = r
  
      # If root is not largest, swap with largest and continue heapifying
      if largest != i:
          arr[i], arr[largest] = arr[largest], arr[i]
          heapify(arr, n, largest)
  
  
  def heapSort(arr):
      n = len(arr)
  
      # Build max heap
      for i in range(n//2, -1, -1):
          heapify(arr, n, i)
  
      for i in range(n-1, 0, -1):
          # Swap
          arr[i], arr[0] = arr[0], arr[i]
  
          # Heapify root element
          heapify(arr, i, 0)
  
  
  arr = [1, 12, 9, 5, 6, 10]
  heapSort(arr)
  n = len(arr)
  print("Sorted array is")
  for i in range(n):
      print("%d " % arr[i], end='')
```
  
