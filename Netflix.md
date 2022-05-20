## Feature # 1: We want to enable users to see relevant search results despite minor typos.
- it means that we're given a list of titles (words) and we need to group the titles that are anagrams of eachother (An anagram is a word, phrase, or sentence formed from another word by rearranging its letters.)
- we create a hashMap with 
  - key: a tuple of 26 indices (related to 26 letters in alphabet) and each value in a index corresponds to frequency that letter is used in a word: `tuple(0,1,..,3)`
  - value: list of words that are anagrams (have the same frequency of letters and the same letters)

- `{(0,0..,1..): ['duel', 'dule']}`

```py
def searchSimilarTitles(titles):
    hashMap = {}
    
    for title in titles:
        freq = [0 for _ in range(26)]
        
        for char in title:
            index = ord(char) - ord('a')
            freq[index] += 1
        
        key = tuple(freq)
        
        if key in hashMap:
            hashMap[key].append(title)
        else:
            hashMap[key] = [title]
        
    print(hashMap)
    return list(hashMap.values())

titles = ["duel","dule","speed","spede","deul","cars"]
print(searchSimilarTitles(titles))
```
- `Time complexity: O(n*k)`: Let n be the size of the list of strings, and k be the maximum length that a single string can have.
- `Space complexity: O(n*k)`: we still save all the strings in dictionary
