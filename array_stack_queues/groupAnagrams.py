def groupAnagrams(strs):
    d = {}
    print(sorted(strs))
    for w in strs:
        key = tuple(sorted(w))
        # print(key)
        d[key] = d.get(key, []) + [w]
    print(d)
    return d.values()


groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"])

https: // leetcode.com/explore/other/card/30-day-leetcoding-challenge/528/week-1/3288/
# Approach 2: Categorize by Count
Intuition

Two strings are anagrams if and only if their character counts(respective number of occurrences of each character) are the same.

Algorithm

We can transform each string \text{s}s into a character count, \text{count}count, consisting of 26 non-negative integers representing the number of \text{a}a's, \text{b}b's, \text{c}c's, etc. We use these counts as the basis for our hash map.

# 1#2#3#0#0#0...#0 where there are 26 entries total. In python, the representation will be a tuple of the counts. For example, abbccc will be (1, 2, 3, 0, 0, ..., 0), where again there are 26 entries total.
In Java, the hashable representation of our count will be a string delimited with '#' characters. For example, abbccc will be

Anagrams


Complexity Analysis

Time Complexity: O(NK)O(NK), where NN is the length of strs, and KK is the maximum length of a string in strs. Counting each string is linear in the size of the string, and we count every string.

Space Complexity: O(NK)O(NK), the total information content stored in ans.
