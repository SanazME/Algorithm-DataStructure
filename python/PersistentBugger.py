"""
Write a function, persistence, that takes in a positive parameter num and returns its multiplicative persistence, which is the number of times you must multiply the digits in num until you reach a single digit.

For example:

 persistence(39) => 3  # Because 3*9 = 27, 2*7 = 14, 1*4=4
                       # and 4 has only one digit.
"""
" %run ./my_script.py"

#My solutin
def persistence(n):
    counter = 0
    return helper_persistance(n, counter)

def helper_persistance(num, counter):
    digits = []
    product = 1
    while (num//10 != 0):
        digits.append(num%10)
        product *= num%10
        num = num//10
    product *= num
    digits.append(num)
    if len(digits) != 1:
        counter += 1
        return helper_persistance(product, counter)
    else:
        return counter


# Best solutin
def persistence_better(n):
    import functools
    counter = 0
    while n >= 10:
        n=functools.reduce(lambda a,b: a*b, [int(x) for x in str(n)])

        counter += 1
