### Buffered File:
- Assume we have a File class whose constructor takes a filepath string as an argument. It has a single method called write, which persists bytes directly to disk.
```py
# f = File('/tmp/my/file.txt')
# f.write(b"hello world")
```
Write a wrapper class for the file object which allows us to buffer the writes in-memory. The wrapper class, BufferedFile is initialized with a File class object and a buffer size. It has two methods: write and flush. The data should be flushed to disk when the buffer is full, or on demand with a method called flush. All bytes must be stored in the buffer first before being written to disk. The buffer cannot use more memory than the max bytes allowed.


**Solution**
- A suitable datastructure for buffer is `bytearray(<size>)`:  a mutable sequence of bytes and can be easily converted to byte `bytes(<the array>)` for write on disk. We can initialized a fixed-length buffer and use two pointers (indices):
  -  `write`: to keep track of where the next byte should be written
  -  `read` : to keep track of where to start reading when flushing to disk (meaning the previous bytes were already flushed to disk)

- **Buffer state:**
  - empty buffer is when the read and write indices are the same: `reader == writer`
  - full buffer is when the next byte to write is equal to reader so if write the next byte, we overwrite the data: `(writer + 1) % buffer_size == reader`
  - available space:
      - if write_pointer >= read_pointer:
        - `[---RXXXXXW---]`, where `-` is available space, `X` is data hasn't been flushed yet.
        - so the available space is the total length - (write - read)
        - `[XXXW-----RXXX]`, where `-` is available space, `X` is data hasn't been flushed yet.
        - so the available space is the (write - read)
      - it's a cricular buffer so for available space, when we subtract writer from buffer_size, we also need to add reader posotion because we can overwrite elements that were alreadt read and flushed to disk so that space is also available.
   
```py
 class File:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def write(self, bytes: bytearray):
        with open(self.file_path) as f:
            f.write(bytes)

class BufferedFile:
    def __init__(self, file_obj, buffer_size):
        self.file = file_obj
        self.buffer_size = buffer_size
        self.buffer = bytearray(buffer_size)
        self.read_pointer = 0
        self.write_pointer = 0

    def write(self, data):
        bytes_written = 0
        while bytes_written < len(data):
            available_space = self._available_space()
            if available_space == 0:
                self.flush()
                available_space = self.buffer_size

            chunk_size = min(available_space, len(data) - bytes_written)
            end_pos = (self.write_pointer + chunk_size) % self.buffer_size

            if end_pos > self.write_pointer:
                self.buffer[self.write_pointer:end_pos] = data[bytes_written:bytes_written+chunk_size]
            else:
                first_part = self.buffer_size - self.write_pointer
                self.buffer[self.write_pointer:] = data[bytes_written:bytes_written+first_part]
                self.buffer[:end_pos] = data[bytes_written+first_part:bytes_written+chunk_size]

            self.write_pointer = end_pos
            bytes_written += chunk_size

    def flush(self):
        if self.read_pointer < self.write_pointer:
            self.file.write(bytes(self.buffer[self.read_pointer:self.write_pointer]))
        else:
            self.file.write(bytes(self.buffer[self.read_pointer:]))
            self.file.write(bytes(self.buffer[:self.write_pointer]))
        self.read_pointer = self.write_pointer

    def _available_space(self):
        if self.write_pointer >= self.read_pointer:
            return self.buffer_size - (self.write_pointer - self.read_pointer)
        else:
            return self.read_pointer - self.write_pointer

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.flush()
```

with `__enter__` and `__exit__` we can use `with` for our wrapped buffer file like:
```py
with BufferedFile(file_obj, buffer_size) as bf:
    bf.write(some_data)
    bf.write(more_data)
# The file is automatically flushed here when exiting the 'with' block
```

### Quert log processing
You are given a stream of log entries and queries. Your task is to process these entries and generate an output that matches logs with relevant queries. The system should work as follows:
- Process input strings that start with either "Q:" (queries) or "L:" (logs).
- For each query (Q:), generate an acknowledgment (ACK:) with a unique ID.
- For each log (L:), check if it matches any of the previously seen queries.
- A log matches a query if all the words in the query are present in the log (case-insensitive, order doesn't matter).
- Generate an output for each matching log (M:) along with the IDs of all matching queries.
```py
livetail_stream = [
  "Q: database",
  "Q: Stacktrace",
  "Q: loading failed",
  "L: Database service started",
  "Q: snapshot loading",
  "Q: fail",
  "L: Started processing events",
  "L: Loading main DB snapshot",
  "L: Loading snapshot failed no stacktrace available",
]

livetail_output = [
  "ACK: database; ID=1",
  "ACK: Stacktrace; ID=2",
  "ACK: loading failed; ID=3",
  "M: Database service started; Q=1",
  "ACK: snapshot loading; ID=4",
  "ACK: fail; ID=5",
  "M: Loading main DB snapshot; Q=4",
  "M: Loading snapshot failed no stacktrace available; Q=2,3,4",
]
```

**Solution**
- Solution 1 is simple and uses less memory. However, `query_words.issubset(log_words)` in the worst case can have time complexity of `O(Q * W)` where `Q` is the number of queries and `W` is the average number of words in a query. 

- In the second solution, Improved matching algorithm: We first find the intersection of all query IDs that match any word in the log. Then we filter this set to ensure all words from each query are present in the log.
- 
```py
from collections import defaultdict

def process_livetail(stream):
    query_id = 0
    queries = {}
    word_to_queries = defaultdict(set)
    output = []

    def process_query(content):
        nonlocal query_id
        query_id += 1
        words = set(content.lower().split())
        queries[query_id] = words
        for word in words:
            if word not in word_to_queries:
                 word_to_queries[word] = set()
            word_to_queries[word].add(query_id)
        output.append(f"ACK: {content}; ID={query_id}")

    def process_log(content):
        log_words = set(content.lower().split())
        intersec_queries = set()
        for word in log_words:
            if word in word_dic:
                intersec_queries.update(word_dic[word])

        matching_queries = {str(queryId) for queryId in intersec_queries if query_dic[queryId].issubset(log_words)}
        if matching_queries:
            //query_ids = ",".join(map(str, sorted(matching_queries)))
            query_ids = ",".join(sorted(matching_queries))
            output.append(f"M: {content}; Q={query_ids}")

    for entry in stream:
        entry_type, content = entry.split(":", 1)
        content = content.strip()
        if entry_type == "Q":
            process_query(content)
        elif entry_type == "L":
            process_log(content)

    return output

# Test the function
livetail_stream = [
    "Q: database",
    "Q: Stacktrace",
    "Q: loading failed",
    "L: Database service started",
    "Q: snapshot loading",
    "Q: fail",
    "L: Started processing events",
    "L: Loading main DB snapshot",
    "L: Loading snapshot failed no stacktrace available",
]

result = process_livetail(livetail_stream)
for line in result:
    print(line)
```

**Solution 1**
```py
def process(stream):    
    def processQuery(content):
        nonlocal query_id
        query_id += 1
        
        query_words = set(content.lower().split())
        query_dic[query_id] = query_words
        
        
        return f"ACK: {content}; ID={query_id}"
        
        
    def processLog(content):
        log_words = set(content.lower().split())
        matched_queries = []
        for query_id, query_words in query_dic.items():
            if query_words.issubset(log_words):
                matched_queries.append(str(query_id))

        if matched_queries:
            return f"M: {content}; Q={','.join(matched_queries)}" 
        
        else:
            return
        
     
    output = []
    query_id = 0
    query_dic = {}
    for entry in stream:
        content_type, content = entry.split(":", 1)
        content = content.strip()
        if content_type == "Q":
            out = processQuery(content)
            print(out)
            output.append(out)
        elif content_type == "L":
            out = processLog(content)
            print(out)
            output.append(out)
            
        
    return output
```
### Latencies in bucket ranges:
Question 1 - Given array of latencies[], sort them in bucket ranges.
ranges - 0-9, 10-19, 20-29, 30-39, 40-49,50-59, 60-69, 70-79, 80-89, 90-99, 100>=

Input: latencies[6,7,50, 100,110]
output:
0-9 - 2
50-59 - 1
100 - 2


### Files and subsirectories
Question 2: Given root directory find the total sizes for the files in the sub-directories.


### Screen : count words repetition + DFS type
Coding-1 : buffered file + follow ups (can be found in others DD posts)



### High Perf filtering
There is a stream that has coming tags and also has a list of keywords, design a high performance filter to output these keywords remaining tags.
For example: given stream `['apple, facebook, google', 'banana, facebook', 'facebook, google, tesla', 'intuit, google, facebook']`, if the keyword is `['apple']` the output should `['facebook', 'google']` because only `'apple, facebook, google'` has `apple`. Similarly if the keyword is `['facebook', 'google']`, the output should `['apple', 'tesla', 'intuit']`. The output can be in any order and can be put into a single list/array.

I was not sure how to handle these:

High performance filter.
The tags are coming as in a stream.

maybe similar to: https://leetcode.com/problems/people-whose-list-of-favorite-companies-is-not-a-subset-of-another-list/description/

### Another filetring
"You are given the list of logs of HTTP requests in the given format:

`[IP, HTTP Verb, status, response time, size, request time]`
Write the code to answer the various queries. For example, list all HTTP requests for the last 2 months or show all requests with 200 status, etc."


### Rm -rf
Assuming you have these functions:
```py
Delete(path) -> bool: deletes the file or empty directory
IsDirectory(path) -> bool: checks whether filepath is directory or not
GetAllFiles(path) -> List<string>: gets the absolute paths of all files in a directory, including other directories
```
Implement rm -rf.
```py
define DeleteAllFilesAndDir(path):
```

How do you code it in a way that prevents out of memory (OOM) errors?
