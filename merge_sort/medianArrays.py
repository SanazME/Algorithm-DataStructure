"""
Find the median of two sorted arrays

arr1 = [1,3,5]
arr2 = [2,4,6]

median(arr1,arr2)=3.5

"""

def median(arr1, arr2):
    num_elems = len(arr1)+len(arr2)
    # Location of the median element in the stiched list
    median_elem = num_elems//2+1

    i=j=last=prev=0

    if len(arr1)==0 and len(arr2)==0:
        return 

    else:
        while median_elem:
            if i<len(arr1) and j<len(arr2):
                if arr1[i]<arr2[j]:
                    last=arr1[i]
                    i+=1
                else:
                    last=arr2[j]
                    j+=1
            elif i<len(arr1):
                last = arr1[i]
                i+=1
            elif j<len(arr2):
                last=arr2[j]
                j+=1
                    
            median_elem -=1
            if median_elem > 0:
                prev = last

        if num_elems%2:
            return last
        else:
            return (last+prev)/2

print(median([1,3,6],[3,6,8]))

