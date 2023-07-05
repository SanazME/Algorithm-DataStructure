## Find Median given frequency of Numbers

Schema:
```sql
Create table If Not Exists Numbers (num int, frequency int)
Truncate table Numbers
insert into Numbers (num, frequency) values ('0', '7')
insert into Numbers (num, frequency) values ('1', '1')
insert into Numbers (num, frequency) values ('2', '3')
insert into Numbers (num, frequency) values ('3', '1')
```

### Answer:
- suppose `x` has a freq of `n` and
- total freq of other numbers that on its left is `l` : how many numbers are smaller than x
- on its right is `r`: how many numbers of larger than x.
- The difference is : `(n + l) - (n + r) = |l - r|`:
  - if the diff is zero, `x` is the median
  - if the diff is not zero but `n` can cover the diff, `x` is the median.

consider the case where L and N is not the same length (how many numbers)
only case the n is the median is that when add N to to smaller side of L and R, that side become the bigger side
when L < R, add N to smaller side L become L + N > R
when L > R, add N to smaller side R become L < R + N

if n is not the median, then L < R, add N to the small side still L + N < R, means N is not in the middle

what if L = R, then we know L are count of how many numbers small than n, R is how many numbers bigger than n, then n is the median already
so hence the query is
N >= abs( ( L + N ) - (R + N) )
 
```sql
select avg(n.num) as median
from Numbers as n
Where n.frequency >= abs(
(select sum(frequency) from Numbers where num<=n.num) -
(select sum(frequency) from Numbers where num>=n.num)
)
```  
