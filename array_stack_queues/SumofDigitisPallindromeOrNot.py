
# Get the number of test cases
test_numbers= input()
list_results=[]
for n in range(int(test_numbers)):
    number = int(input())
    remainder = 0
    while number:
        remainder, number = remainder + number%10, number//10
    if len(str(remainder)) <=1:
        list_results.append(True)
    elif (remainder%10)==(remainder//10):
        list_results.append(True)
    else:
        list_results.append(False)
        




    list_numbers.append(int(number))
