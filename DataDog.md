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

**Initial Solution**
```py
def sortLatency(latencies):
    buckets = [x for x in range(0,101,10)]
    freq = defaultdict(list)
    binNameMap = {
        0 : "0-9",
        10 : "10-19",
        20 : "20-29",
        30 : "30-39",
        40 : "40-49",
        50 : "50-59",
        60 : "60-69",
        70 : "70-79",
        80 : "80-89",
        90 : "90-99",
        100 : "100",
        
    }
   
    def binarySearch(nums, target):
        if len(nums) == 0:
            return -1
        left, right = 0, len(nums) - 1
        
        while left <= right:
            mid = (left + right) // 2
            # print(target, nums[mid])
            
            if nums[mid] == target:
                return nums[mid]
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1

        if right < 0:
            return nums[0]
        elif left >= len(nums):
            return nums[-1]
        else:
            return nums[right]
    
    for ele in latencies:
        binVal = binarySearch(buckets, ele)
        freq[binVal] = freq.get(binVal, 0) + 1
    
    result = ""
    for key in freq:
        result += f"{binNameMap[key]} - {freq[key]}\n"
        
    return result
```


**Follow-up questions**
a) How would you modify the solution to handle custom bucket ranges that aren't evenly spaced?
b) Can you implement this solution using a streaming approach, processing one number at a time?
e) How would you modify the code to make it more memory-efficient for very large input arrays?

**a) Handle custom buckets**
```py
def sortLatency(latencies, custom_buckets):
    
    def binarySearch(nums, target):
        if len(nums) == 0:
            return -1
        
        left, right = 0, len(nums) - 1
        
        while left <= right:
            mid = (left + right) // 2
            
            if nums[mid] == target:
                return nums[mid]
            elif nums[mid] > target:
                right = mid - 1
            else:
                left = mid + 1
                
        if right < 0:
            return nums[0]
        elif left >= len(nums):
            return nums[-1]
        else:
            return nums[right]
        
    freq = defaultdict(list)
    for latency in latencies:
        bucketVal = binarySearch(custom_buckets, latency)
        print(f"{latency} belongs in bucket: {bucketVal}")
        freq[bucketVal] = freq.get(bucketVal, 0) + 1
        
    result = ""
    for i in range(len(custom_buckets)):
        if i == len(custom_buckets) - 1:
            bucketName = f"{custom_buckets[i]}"
        else:
            bucketName = f"{custom_buckets[i]}-{custom_buckets[i+1] - 1}"
        result += f"{bucketName} - {freq[custom_buckets[i]]}\n"
        
    return result
        
custom_buckets = [0, 10, 25, 50, 100, 200]
latencies = [6, 7, 50, 100, 110, 15, 30, 75, 190, 250]
print(sortLatency(latencies, custom_buckets))
```

**e) more memeory efficient**
- user generator to pass in latencies one at a time to the function instead of passing a too large list of latencies. Also, we use integer division instead of binary search for our fixed lenght buckets:
```py
def latency_generator():
    yield from [6, 7, 50, 100, 110, 15, 30, 75, 190, 250]

def sortLatency(latencies):
    
    def bucketName(latency):
        if latency >= 100:
            return 100
        else:
            return (latency // 10) * 10
        
    freq = defaultdict(list)
    for latency in latencies:
        bucketVal = bucketName(latency)
        print(f"{latency} belongs in bucket: {bucketVal}")
        freq[bucketVal] = freq.get(bucketVal, 0) + 1
        
    result = ""
    keySorted = sorted(freq.keys())
    for bucket in keySorted:
        if bucket == 100:
            bucketName = "100"
        else:
            bucketName = f"{bucket}-{bucket + 10 - 1}"
        result += f"{bucketName} - {freq[bucket]}\n"
        
    return result
        
custom_buckets = [0, 10, 25, 50, 100, 200]
latencies = [6, 7, 50, 100, 110, 15, 30, 75, 190, 250]
print(sortLatency(latency_generator()))
```
**b) streaming solution**
we persist the information with streaming and so anytime the user asks for results, we can return the results up to now that we have processed:
```py
class StreamingLatencySorter:
    def __init__(self):
        self.freq = defaultdict(list)
        self.bin_name_map = {
            i : f"{i}-{i+9} " for i in range(0,100,10)
        }
        self.bin_name_map[100] = "100"
        
    def bucketName(self, latency):
        if latency >= 100:
            return 100
        else:
            return (latency // 10) * 10
        
    def process(self, latency):
        bucket = self.bucketName(latency)
        self.freq[bucket] = self.freq.get(bucket, 0) + 1
        
    def getResult(self):
        result = ""
        for key in sorted(self.freq.keys()):
            result += f"{self.bin_name_map[key]} - {self.freq[key]}\n"
            
        return result
            
streamer = StreamingLatencySorter()    
latencies = [6, 7, 50, 100, 110, 15, 30, 75, 190, 250]
for latency in latencies:
    streamer.process(latency)
    
print(streamer.getResult())
```

### Files and subsirectories
Question 2: Given root directory find the total sizes for the files in the sub-directories.

```py
import os

def get_directory_size(path):
    total_size = 0
    
    # Iterate over all items in the directory
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        
        if os.path.isfile(item_path):
            # Base case: If it's a file, add its size
            total_size += os.path.getsize(item_path)
        elif os.path.isdir(item_path):
            # Recursive case: If it's a directory, recurse into it
            total_size += get_directory_size(item_path)
    
    return total_size

# Usage
root_dir = "/path/to/root/directory"
total_size = get_directory_size(root_dir)
print(f"Total size: {total_size} bytes")
```
- However, if there are many recursive dirctories, we get OOM (out of memory) from recursive solution because the recursive function holds the space till nested functions are completed and if there are too many nested function calls, it will eventually OOM. Instead let's use iterative approach:

```py
import os

def get_directory_size(path):
    stack = [path]
    total_size = 0
    
    while stack:
        current_path = stack.pop()
        
        try:
            # If it's a file, add its size
            if os.path.isfile(current_path):
                total_size += os.path.getsize(current_path)
            # If it's a directory, add its contents to the stack
            elif os.path.isdir(current_path):
                with os.scandir(current_path) as it:
                    stack.extend(entry.path for entry in it)
        except PermissionError:
            print(f"Permission denied: {current_path}")
        except FileNotFoundError:
            print(f"File or directory not found: {current_path}")
        except Exception as e:
            print(f"Error processing {current_path}: {e}")
    
    return total_size

# Usage
root_dir = "/path/to/root/directory"
total_size = get_directory_size(root_dir)
print(f"Total size: {total_size} bytes")
```
Another solution that uses `os.DirEntry` methods:
- Loop structure: Instead of extending the stack with all entries at once, we process them one by one. This can be more memory-efficient for large directories.
- File size calculation: Use entry.stat().st_size instead of os.path.getsize(). This is more efficient as it doesn't require another system call.
- Directory handling: When we encounter a directory, we push a new scandir object onto the stack instead of its contents.
```py
import os

def get_directory_size(path):
    if not path or not os.path.exists(path):
        return 0
    
    total_size = 0
    stack = [os.scandir(path)]
    
    while stack:
        try:
            entry = next(stack[-1])   # because os.scandir(path) returns an iterator
            if entry.is_file(follow_symlinks=False):
                total_size += entry.stat().st_size
            elif entry.is_dir(follow_symlinks=False):
                stack.append(os.scandir(entry.path))
        except StopIteration:
            stack.pop()
        except PermissionError:
            print(f"Permission denied: {entry.path}")
        except FileNotFoundError:
            print(f"File or directory not found: {entry.path}")
        except Exception as e:
            print(f"Error processing {entry.path}: {e}")
    
    return total_size

# Usage
root_dir = "/path/to/root/directory"
total_size = get_directory_size(root_dir)
print(f"Total size: {total_size} bytes")
```
**Follow up questions**
**1. How would you modify the function to return a dictionary mapping each subdirectory to its total size?**
```py
import os
from collections import defaultdict

def get_directory_size_mapping(root_path):
    if not root_path or not os.path.exists(root_path):
        return {}

    size_mapping = defaultdict(int)
    stack = [(root_path, os.scandir(root_path))]
    
    while stack:
        current_path, it = stack[-1]
        try:
            entry = next(it)
            full_path = entry.path
            if entry.is_file(follow_symlinks=False):
                try:
                    file_size = entry.stat().st_size
                    size_mapping[current_path] += file_size
                    # Update all parent directories
                    parent = os.path.dirname(current_path)
                    while parent and parent >= root_path:
                        size_mapping[parent] += file_size
                        parent = os.path.dirname(parent)
                except (PermissionError, FileNotFoundError):
                    print(f"Error accessing file: {full_path}")
            elif entry.is_dir(follow_symlinks=False):
                stack.append((full_path, os.scandir(full_path)))
        except StopIteration:
            stack.pop()
        except PermissionError:
            print(f"Permission denied: {current_path}")
            stack.pop()
        except Exception as e:
            print(f"Error processing {current_path}: {e}")
            stack.pop()

    return dict(size_mapping)

# Usage
root_dir = "/path/to/root/directory"
size_map = get_directory_size_mapping(root_dir)
for directory, size in size_map.items():
    print(f"{directory}: {size} bytes")
```
- We use a defaultdict(int) to store the size mapping. This automatically initializes a new key with 0 when we first access it.
- The stack now contains tuples of (current_path, scandir_iterator) instead of just paths. This allows us to keep track of which directory we're currently processing.
- At the end, we convert the defaultdict to a regular dictionary before returning.


**2. How can you optimize the solution for very large directory structures?**
There are several strategies we can use to improve performance and reduce memory usage:
1. Multithreading: we can use `concurrent.futures` to scan directories in parallel
2. Batch processing, instead of adding individual tasks for each directory, we can process directories in batches.

```py
import os
import concurrent.futures
import threading
from collections import defaultdict

def scan_directory(path):
    total_size = 0
    file_count = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks=False):
                total_size += entry.stat().st_size
                file_count += 1
    except PermissionError:
        print(f"Permission denied: {path}")
    except Exception as e:
        print(f"Error scanning {path}: {e}")
    return path, total_size, file_count

def get_directory_size_optimized(root_path, max_threads=os.cpu_count()):
    if not root_path or not os.path.exists(root_path):
        return {}

    size_mapping = defaultdict(lambda: [0, 0])  # [total_size, file_count]
    dir_queue = [root_path]
    lock = threading.Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        while dir_queue:
            batch = []
            for _ in range(min(len(dir_queue), max_threads * 2)):  # Process in batches
                if dir_queue:
                    batch.append(dir_queue.pop(0))

            futures = [executor.submit(scan_directory, dir_path) for dir_path in batch]

            for future in concurrent.futures.as_completed(futures):
                dir_path, dir_size, file_count = future.result()
                with lock:
                    size_mapping[dir_path][0] += dir_size
                    size_mapping[dir_path][1] += file_count
                    
                    # Update parent directories
                    parent = os.path.dirname(dir_path)
                    while parent and parent >= root_path:
                        size_mapping[parent][0] += dir_size
                        size_mapping[parent][1] += file_count
                        parent = os.path.dirname(parent)

                # Add subdirectories to the queue
                try:
                    dir_queue.extend(entry.path for entry in os.scandir(dir_path) 
                                     if entry.is_dir(follow_symlinks=False))
                except PermissionError:
                    print(f"Permission denied: {dir_path}")
                except Exception as e:
                    print(f"Error processing {dir_path}: {e}")

    return {k: {'size': v[0], 'file_count': v[1]} for k, v in size_mapping.items()}

# Usage
root_dir = "/path/to/large/directory"
result = get_directory_size_optimized(root_dir)
for directory, info in result.items():
    print(f"{directory}: Size: {info['size']} bytes, Files: {info['file_count']}")
```

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
