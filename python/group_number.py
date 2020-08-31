"""
The city of Darkishland has a strange hotel with infinite rooms. The groups that come to this hotel follow the following rules:

At the same time only members of one group can rent the hotel.

Each group comes in the morning of the check-in day and leaves the hotel in the evening of the check-out day.

Another group comes in the very next morning after the previous group has left the hotel.

A very important property of the incoming group is that it has one more member than its previous group unless it is the starting group. You will be given the number of members of the starting group.

A group with n members stays for n days in the hotel. For example, if a group of four members comes on 1st August in the morning, it will leave the hotel on 4th August in the evening and the next group of five members will come on 5th August in the morning and stay for five days and so on.

Given the initial group size you will have to find the group size staying in the hotel on a specified day.

Input
S denotes the initial size of the group and D denotes that you will have to find the group size staying in the hotel on D-th day (starting from 1). A group size S means that on the first day a group of S members comes to the hotel and stays for S days. Then comes a group of S + 1 members according to the previously described rules and so on.

Test.assert_equals(group_size(1, 6), 3)
Test.assert_equals(group_size(3, 10), 5)
Test.assert_equals(group_size(3, 14), 6)

"""

# 1st group spends in the hotel s days,
#  2nd group - s + 1 days,
#  3rd group - s + 2 days,
#  ...,
#  nth group - s + n - 1 days.
#
# The total number of days for n groups: n * (s + s + n - 1) / 2 
#  (by the formula of arithmetic series).
# Let d be the last day of the nth group. Then
#  n * (s + s + n - 1) / 2 = d, 
#  n**2 + (2*s-1)*n - 2*d = 0,  
#  The only possible root of this quadratic equation equals
#  1/2 * (-p + sqrt(p**2 - 4*q), where p = 2*s - 1, q = 2*d.  
#  Thus, n = (1 - 2*s + sqrt((2*s - 1)**2 + 8*d)) / 2.
# But if d is not the last day of the group n, then n will be fractional,
#   and we have to round it up (get the ceiling of n).

def group_size(s, d):
    n = ceil((1 - 2*s + sqrt((2*s - 1)**2 + 8*d)) / 2)
    return s + n - 1