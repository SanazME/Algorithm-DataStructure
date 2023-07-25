## 1. Longest Palindromic Substring
- https://leetcode.com/problems/longest-palindromic-substring/
```py
def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """           
        if len(s) <= 1:
            return s
    
        soFar = ''
        for i, char in enumerate(s):
            # for odd case: 'aba'
            tmp = self.helper(s, i, i)
            if len(tmp) >= len(soFar):
                soFar = tmp

            # for even case: 'abba'
            tmp = self.helper(s, i, i+1)

            if len(tmp) >= len(soFar):
                soFar = tmp

        return soFar
        
        
  def helper(self, s, left, right):
      while left >= 0 and right < len(s) and s[left] == s[right]:
          left -= 1
          right += 1
      return s[left+1:right]
```

## 2. Web Crawler Multithread - Databrick
- https://leetcode.com/problems/web-crawler-multithreaded/description/

```py
# """
# This is HtmlParser's API interface.
# You should not implement it, or speculate about its implementation
# """
#class HtmlParser(object):
#    def getUrls(self, url):
#        """
#        :type url: str
#        :rtype List[str]
#        """
from concurrent import futures
class Solution:
    def crawl(self, startUrl: str, htmlParser: 'HtmlParser') -> List[str]:
        self.hostname = self.extractHostname(startUrl)


        # paramlist = parameters.split("/")

        visited = set()
        visited.add(startUrl)

        # Solution 1
        # queue = deque([])

        # result = [startUrl]

        # with futures.ThreadPoolExecutor(max_workers=8) as executor:
        #     queue.append(executor.submit(htmlParser.getUrls, startUrl))

        #     while queue:
        #         for url in queue.popleft().result():
        #             print("url: ", url)
        #             if self.extractHostname(url) == hostname and url not in visited:
        #                 result.append(url)
        #                 visited.add(url)
        #                 queue.append(executor.submit(htmlParser.getUrls, url))


        # return result


        # Solution 2
        self.htmlParser = htmlParser
        self.pending = []
        self.seen = set()
        self.count = 0

        with futures.ThreadPoolExecutor(max_workers=8) as self.executor:
            self.submitToExecute(startUrl)
        

            while self.pending:
                self.count += 1
                print("count: ", self.count)
                
                pendingSoFar, self.pending = self.pending, []
                
                for ele in futures.as_completed(pendingSoFar):
                    if e := ele.exception():
                        print(e)
                        

        
        return self.seen


    def submitToExecute(self, url):
        self.seen.add(url)

        print("seen: ", self.seen)

        self.pending.append(self.executor.submit(self.processUrl, url))
        
        


    def processUrl(self, url):
        print(self.extractHostname(url), self.hostname)
        urls = set()
        for uul in self.htmlParser.getUrls(url):
            print("url here: ", uul)
            if self.extractHostname(uul) == self.hostname:
                urls.add(uul)
        # urls = set(url for urls in self.htmlParser.getUrls(url) if self.extractHostname(url) == self.hostname)
        # print("urls: ", urls)
        for url in urls - self.seen:
            self.submitToExecute(url)






    def extractHostname(sefl, url):
        if url.startswith("http://"):
            url = url[7:]
        elif url.startswith("https://"):
            url = url[8:]

        end = url.find("/")

        if end == -1:
            end = len(url)

        else:
            port  = url.find(":")
            if port != -1 and port < end:
                end = port

        hostname = url[:end]
        parameters = url[end:]

        return hostname
```

```java
/**
 * // This is the HtmlParser's API interface.
 * // You should not implement it, or speculate about its implementation
 * interface HtmlParser {
 *     public List<String> getUrls(String url) {}
 * }
 */
class Solution {
    public List<String> crawl(String startUrl, HtmlParser htmlParser) {
        String hostname = getHostname(startUrl);

        List<String> result = new ArrayList<>();
        Set<String> visited = new HashSet<>();

        BlockingQueue<String> queue = new LinkedBlockingQueue<>();
        Deque<Future> tasks = new ArrayDeque<>();

        queue.offer(startUrl);

        ExecutorService executor = Executors.newFixedThreadPool(4, r -> {
            Thread t = new Thread(r);
            t.setDaemon(true);
            return t;
        });

        while (true){
            String url = queue.poll();
            if (url != null) {
                if (getHostname(url).equals(hostname) && !visited.contains(url)){
                    result.add(url);
                    visited.add(url);

                    tasks.add(executor.submit(() -> {
                        List<String> newUrls = htmlParser.getUrls(url);
                        for (String newUrl: newUrls){
                            queue.offer(newUrl);
                        }
                    }
                    ));

                }
            } else {
                if (!tasks.isEmpty()){
                    Future nextTask = tasks.poll();

                    try {
                        nextTask.get();
                    } catch (InterruptedException | ExecutionException e) {}

                } else {
                    break;
                }
            }
        }

        return result;
        
    }



    private String getHostname(String url){
        if (url.startsWith("https://")){
            return url.substring(8).split("/")[0];
        } else if (url.startsWith("http://")){
            return url.substring(7).split("/")[0];
        }
        return url.substring(7).split("/")[0];
    }
}
```
