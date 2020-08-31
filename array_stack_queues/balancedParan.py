"""
(coderbyte)
Generate all balanced bracket combinations
For this popular algorithm interview question, your goal is to print all possible balanced parenthesis combinations up to N. For example:

N = 2
(()), ()()

N = 3
((())), (()()), (())(), ()(()), ()()()

Algorithm
  
We will implement a recursive function to solve this challenge. The idea is:

(1) Add a left bracket to a newly created string.
(2) If a left bracket was added, potentially add a new left bracket and add a right bracket.
(3) After each of these steps we add the string to an array that stores all bracket combinations.

This recursive solution might be hard to see at first, but using N = 2 as an example, the steps taken are:

N = 2
create (
create ()

N = 1
create ((
create ()(

N = 0
add (())
add ()()

Done when N = 0

"""
arr=[]
def balancedParan(left,right,string):
    #Base
    if left==0 and right==0:
        arr.append(string)
    if left>0:
        balancedParan(left-1, right+1, string+"(")
    if right>0:
        balancedParan(left, right-1, string+")")

balancedParan(3,0,"")
print(arr)