
# Get the number of test cases
test_numbers= input()
list_results=[]
for n in range(int(test_numbers)):
    number = int(input())
    remainder = 0
    while number:
        remainder, number = remainder + number%10, number//10
    if len(str(remainder)) <=1:
        list_results.append('YES')
    elif (remainder%10)==(remainder//10):
        list_results.append('YES')
    else:
        list_results.append('NO')
for item in list_results:
    print(item)
