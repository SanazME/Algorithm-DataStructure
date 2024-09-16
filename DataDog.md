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
  - available space: `(buffer_size - writer + reader) % buffer_size`
      - it's a cricular buffer so for available space, when we subtract writer from buffer_size, we also need to add reader posotion because we can overwrite elements that were alreadt read and flushed to disk so that space is also available.
   
```py
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
