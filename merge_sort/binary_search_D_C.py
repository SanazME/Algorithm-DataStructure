def binary_search_D_C(list, item):
        if len(list)==0:
            return False
        else:
            mid = len(list)/2
            guess = list[mid]
            if item == guess:
                return True
            else:
                if item > guess:
                    return binary_search_D_C(list[mid+1:], item)
                else:
                    return binary_search_D_C(list[:mid], item)
