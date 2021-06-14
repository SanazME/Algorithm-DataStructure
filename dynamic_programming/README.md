## Articles and exercises


- **Tech Dose (youtube):** https://www.youtube.com/c/TECHDOSE4u/search?query=dynamic%20programming
- **strating point:** https://leetcode.com/problems/min-cost-climbing-stairs/solution/
- **Dynamic Programming patterns:** https://leetcode.com/discuss/general-discussion/458695/Dynamic-Programming-Patterns
- **ABCs of Greedy:** https://leetcode.com/discuss/general-discussion/1061059/ABCs-of-Greedy
- https://leetcode.com/discuss/interview-question/815454/Amazon-OA-question-or-SDE-1
- https://leetcode.com/problems/largest-rectangle-in-histogram/
- **Must do Dynamic programming Category wise:** https://leetcode.com/discuss/general-discussion/1050391/Must-do-Dynamic-programming-Problems-Category-wise


## Memoization
- Interleaving string (https://leetcode.com/problems/interleaving-string/). 
- Explanation: https://www.youtube.com/watch?v=EzQ_YEmR598
- We use sliding pointer technique. We use 3 pointers for s1, s2 and s3 strings.
1. if s1 isi divided into N substrings, s2 should be divided into N or N-1 or N+1 substrings
2. len(s3) == len(s1) + len(s2)
3. The order of substrings should be maintained in s3 (alternating)
4. The number of unique characters in s3 should be equal to number of that character in s1 + s2
5. With sliding poniter technique, we define 3 pointes for each s1, s2, s3 strings and when p3 moves on each character in s3, there are 2 options: it can belong to s1 or s2. and so we can create a tree of choices starting from the first character of s3 as the root and each node has two children (each edge corresponds to s1 and s2).
6. With this approach, the time complexity: since each postion has two option times the number of chars in s1 and s2: 2^(m+n). The space complexity is from the recursion call tree depth == len(s3) == m+n
7. to improve it, we can use memoization to remember and save the result of a calculation and for that we use map to save key-value pairs. keys should be the unique states and so one way to create unique keys from three pointers p1, p2 and p3: key = p1 + * + p2 + * + p3 and the value will be whether 

- In memoization version of pascal number or fibonacci, the **Time complexity is O(N) and the space complexity if O(N)**. Without memoization, the time complexity would be **O(2^N)**.
