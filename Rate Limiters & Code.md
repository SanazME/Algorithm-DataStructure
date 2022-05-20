# Rate Limiter
- https://dev.to/satrobit/rate-limiting-using-the-sliding-window-algorithm-5fjn
- https://medium.com/@saisandeepmopuri/system-design-rate-limiter-and-data-modelling-9304b0d18250

**Problem**: Design rate limiter — A rate limiter is a tool that monitors the number of requests per a window time a service agrees to allow. If the request count exceeds the number agreed by the service owner and the user (in a decided window time), the rate limiter blocks all the excess calls(say by throwing exceptions). The user can be a human or any other service(ex: in a micro service based architecture)

**Example:** A device with a specific ip address can only make 10 ticket bookings per minute to a show booking website. Or a service A can hit service B with a rate of at most 500 req/sec. All the excess requests get rejected.


For these use cases, using **persistent memory stores like mysql is a bad idea because the time taken for disk seeks is high enough to hamper the rate limiter granularity**. For instance, let’s say that we’re using a single mysql instance to store the request counts and **mysql takes 1ms to process 1 request**, which means we can achieve a **throughput of 1000 req/sec**. But a rate limiter using in-memory cache takes around **100nanosec(a main-memory access)** to process 1 request, which implies a granularity of around **10Mreq/sec** can be achieved.

**Various Levels:** There are different levels at which one can design rate limiters. Listing a few…

- Rate limiter in a single machine, single threaded scenario
- Rate limiter in a single machine, multi threaded scenario — Handling race conditions
- Rate limiter in a distributed scenario — Distributed Cache Usage like redis
- Rate limiter from client side —Prevent network calls from client to server for all the excess requests.

### 1. Fixed Window counters
- For example, a rate limiter for a service that allows only 10 requests per an hour will have the data model like below. Here, the buckets are the windows of one hour, with values storing the counts of the requests seen in that hour.
- Only the counts of the current window are stored and older windows are deleted when a new window is created(i.e in the above case, if the hour changes, older bucket is deleted)
```json
{
 "1AM-2AM": 7,
 "2AM-3AM": 8
}
```

- Time complexity : O(1) Get and simple atomic increment operation
- Space complexity : O(1) Storing the current window count

### 2. Sliding Window logs
- For every user, a queue of timestamps representing the times at which all the historical calls have occurred within the timespan of recent most window is maintained.

- Time Complexity O(Max requests seen in a time window) - deleting a subset of timestamps (older thatn the start of the current time window)
- Space Complexity O(Max requests seen in a window time) - stores all timestamps of the requests in a time window

**Data Modeling**
**In memory design (Single machine with multiple threads)**
```py
import time
import threading
from collections import deque

class RequestTimeStamps(object):

    # lock is for concurrency in a multi threaded system
	# 100 req/min translates to requests = 100 and windowTimeInSec = 60
    def __init__(self, requests, windowTimeInSec):
        self.requests = requests
        self.windowTimeInSec = windowTimeInSec
        self.timestamps = deque()
        self.lock = threading.Lock()

    # eviction of timestamps older than the window time
    def evictOlderTimestamps(self, currentTimestamp):
        while len(self.timestamps) != 0 and (currentTimestamp - self.windowTimeInSec > self.timestamps[0] ):
            self.timestamps.popleft()



class SlidingWindowLogsRateLimiter(object):

    def __init__(self):
        self.lock = threading.Lock()
        self.rateLimiterMap = {}

    # Default of 100 req/minute
	# Add a new user with a request rate
    def addUser(self, userId, requests=100, windowTimeInSec=60):
        # hold lock to add in the user-metadata map
        with self.lock:
            if userId in self.rateLimiterMap:
                raise Exception("User already present")
            self.rateLimiterMap[usreId] = RequestTimeStamps(requests, windowTimeInSec)

    # Remove a user from the ratelimiter
    def removeUser(self, userId):
        with self.lock:
            if userId in self.rateLimiterMap:
                del self.rateLimiterMap[userId]

    # gives current time epoch in seconds
    @classmethod
    def getCurrentTimestampInSec(cls):
        return int(round(time.time()))


    # Checks if the service call should be allowed or not
    def shouldAllowServiceCall(self, userId):
        with self.lock:
            if userId not in self.rateLimiterMap:
                raise Exception("User is not present.Please whitelist and register user for \
                    service ")
            
            # object RequestTimeStamps
            userTimestamps = self.rateLimiterMap[userId]

            with userTimestamps.lock:
                # current time
                currentTimestamp = self.getCurrentTimestampInSec()
                # remove all the existing older timestamps
                userTimestamps.evictOlderTimestamps(currentTimestamp)
                # Add the current timestamp (better to be a method than directly accessing queue)
                userTimestamps.timestamps.append(currentTimestamp)

                if len(userTimestamps.timestamps) > userTimestamps.requests:
                    return False
                
                return True 
```

### 3. Sliding Window Counter
- This is a hybrid of Fixed Window Counters and Sliding Window logs
- The entire window time is broken down into smaller buckets. The size of each bucket depends on how much elasticity is allowed for the rate limiter
- Each bucket stores the request count corresponding to the bucket range.

For example, in order to build a rate limiter of 100 req/hr, say a bucket size of 20 mins is chosen, then there are 3 buckets in the unit time

For a window time of 2AM to 3AM, the buckets are
```json
{
 "2AM-2:20AM": 10,
 "2:20AM-2:40AM": 20,
 "2:40AM-3:00AM": 30
}
```
If a request is received at 2:50AM, we find out the total requests in last 3 buckets including the current and add them, in this case they sum upto 60 (<100), so a new request is added to the bucket of 2:40AM–3:00AM giving…

**Note:** This is not a completely correct, for example: At 2:50, a time interval from 1:50 to 2:50 should be considered, but in the above example the first 10 mins isn’t considered and it may happen that in this missed 10 mins, there might’ve been a traffic spike and the request count might be 100 and hence the request is to be rejected. But by tuning the bucket size, we can reach a fair approximation of the ideal rate limiter.

- Time Complexity O(1) Fetch the recent bucket, increment and check against the total sum of buckets(can be stored in a totalCount variable).
- Space Complexity O(Number of buckets)

**Data Modeling**
**In memory design (Single machine with multiple threads)**
```py

class RequestCounters(object):
	# Every window time is broken down to 60 parts
	# 100 req/min translates to requests = 100 and windowTimeInSec = 60
	def __init__(self, requests, windowTimeInSec, bucketSize=10):
		self.counts = {}
		self.totalCounts = 0
		self.requests = requests
		self.windowTimeInSec = windowTimeInSec
		self.bucketSize = bucketSize
		self.lock = threading.Lock()

	# Gets the bucket for the timestamp
	def getBucket(self, timestamp):
		factor = self.windowTimeInSec / self.bucketSize
		return (timestamp // factor) * factor

	# Gets the bucket list corresponding to the current time window
	def _getOldestvalidBucket(self, currentTimestamp):
		return self.getBucket(currentTimestamp - self.windowTimeInSec)

	# Remove all the older buckets that are not relevant anymore
	def evictOlderBuckets(self, currentTimestamp):
		oldestValidBucket = self._getOldestvalidBucket(currentTimestamp)
		bucketsToBeDeleted = filter(
			lambda bucket: bucket < oldestValidBucket, self.counts.keys())
		for bucket in bucketsToBeDeleted:
			bucketCount = self.counts[bucket]
			self.totalCounts -= bucketCount
			del self.counts[bucket]


class SlidingWindowCounterRateLimiter(object):
	def __init__(self):
		self.lock = threading.Lock()
		self.ratelimiterMap = {}

	# Default of 100 req/minute
	# Add a new user with a request rate
	# If a request from un-registered user comes, we throw an Exception
	def addUser(self, userId, requests=100, windowTimeInSec=60):
		with self.lock:
			if userId in self.ratelimiterMap:
				raise Exception("User already present")
			self.ratelimiterMap[userId] = RequestCounters(requests, windowTimeInSec)

	def removeUser(self, userId):
		with self.lock:
			if userId in self.ratelimiterMap:
				del self.ratelimiterMap[userId]

	@classmethod
	def getCurrentTimestampInSec(cls):
		return int(round(time.time()))

	def shouldAllowServiceCall(self, userId):
		with self.lock:
			if userId not in self.ratelimiterMap:
				raise Exception("User is not present")

        # object RequestCounters
		userTimestamps = self.ratelimiterMap[userId]

		with userTimestamps.lock:
			currentTimestamp = self.getCurrentTimestampInSec()
			# remove all the existing older timestamps
			userTimestamps.evictOlderBuckets(currentTimestamp)
            # Get current bucket
			currentBucket = userTimestamps.getBucket(currentTimestamp)

			userTimestamps.counts[currentBucket] = userTimestamps.counts.
				get(currentBucket, 0) + 1

			userTimestamps.totalCounts += 1

			if userTimestamps.totalCounts > userTimestamps.requests:
				return False
			return True
```



# Latency Numbers Everyone should know
- https://colin-scott.github.io/personal_website/research/interactive_latency.html

- CPU L1, L2, L3 caches: https://cpuninja.com/cpu-cache/