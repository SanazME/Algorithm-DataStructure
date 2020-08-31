def binary_search_D_C(list, item):
    if len(list) == 0:
        return True
    else:
        first = 0
        last = len(list)-1
        mid = (first+last)/2
        guess = list[mid]
        if item == guess:
            return mid
        else:
            if item > guess:
                return binary_search_D_C(list[mid+1:], item)
                print "new list :", list[mid+1:]
            else:
                return binary_search_D_C(list[0:mid-1], item)
                print "new list: ", list[0:mid-1]
