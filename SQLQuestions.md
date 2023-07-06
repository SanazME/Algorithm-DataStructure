## SQL commands
**`ROW_NUMBER()`**
- it is a window function that assigns a unique sequential number to each row in the result set based on a specified ordering. It can be useful for various purposes, such as ranking rows, pagination, or identifying duplicates. 

```sql
SELECT ROW_NUMBER() OVER (ORDER BY column1, column2, ...) AS row_number, column1, column2, ...
FROM table;
```
In this syntax:

- `ROW_NUMBER()` is the function that generates the sequential numbers.
- `OVER (ORDER BY column1, column2, ...)` defines the ordering of the rows based on one or more columns. The `ORDER BY` clause determines the sequence in which the numbers are assigned.
- `AS` row_number is an optional alias for the generated column, allowing you to refer to it in subsequent parts of the query.
- `column1, column2, ...` represent the columns you want to include in the result set.

When you execute a query using `ROW_NUMBER()`, each row in the result set is assigned a unique number based on the specified ordering. The numbering starts from 1 for the first row and increments by 1 for each subsequent row.

Here's an example to illustrate the usage of `ROW_NUMBER()`:
```sql
SELECT ROW_NUMBER() OVER (ORDER BY salary DESC) AS rank, name, salary
FROM employees;
```
This query generates a result set that includes the rank, name, and salary of each employee. The employees are ordered by their salary in descending order. The ROW_NUMBER() function assigns a sequential rank to each employee based on their salary, with the highest salary receiving rank 1, the next highest receiving rank 2, and so on.

Note that `ROW_NUMBER()` is a window function, which means it operates on a specific subset of rows defined by the OVER clause. You can further customize its behavior using other window functions and clauses such as `PARTITION BY`, `ROWS BETWEEN`

**`PARTITION BY`**
- it is used in conjunction with **window functions** to divide the result set into partitions or groups based on one or more columns. It allows you to apply window functions separately to each partition, enabling more granular and specific calculations within the result set.
```sql
SELECT column1, column2, ..., window_function() OVER (PARTITION BY partition_column1, partition_column2, ...)
FROM table;
```
- `PARTITION BY` specifies the columns based on which you want to partition or group the result set.
- `partition_column1, partition_column2, ...` represent the columns used for partitioning.
- `window_function()` is a window function such as `SUM`, `AVG`, `ROW_NUMBER`, etc. that operates on the partitions.

By using PARTITION BY, you can apply a window function separately to each partition, allowing you to perform calculations and aggregations within specific groups of rows.

Here's an example to illustrate the usage of PARTITION BY:
```sql
SELECT department, employee, salary, AVG(salary) OVER (PARTITION BY department ORDER BY ... ASC/DESC) AS avg_department_salary
FROM employees;
```


## 1. Find Median given frequency of Numbers

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

## 2. Median Employee Salary
- Table: Employee
- Column Names:, id, company, salary
Write an SQL query to find the rows that contain the median salary of each company. While calculating the median, when you sort the salaries of the company, break the ties by id.

### Intuition
- when x is odd, (x + 1)/2 == row_number() we are searching for and when x is even, x/2 and x/2+1 are row_numbers for median.

### Approach
1. Create `view_tmp` with row_number and count window functions which offers info regarding row_num and total_row count
2. select id, company, salary from view_tmp with the condition above.

```sql
with view_tmp AS(
    SELECT id, company, salary,
    ROW_NUMBER() OVER (PARTITION BY company ORDER BY salary asc) row_num,
    COUNT(id) OVER (PARTITION BY company) total_row
    FROM Employee
)
SELECT id, company, salary
FROM view_tmp
WHERE row_num = total_row / 2 or row_num = total_row / 2 + 1 or row_num = (total_row + 1) /2
```
