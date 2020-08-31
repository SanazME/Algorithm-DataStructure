import random
from itertools import accumulate
import math

class RandomGen(object):
    """
    inputs:
        - random_nums : a list of random numbers
        - probabilities : a list of probabilites associated with random numbers
    output : 
        - one of the random_nums element. When this method is called multiple times over a long period, it should return the numbers roughly with the initialized probabilities.
    """
    def __init__(self, random_nums, probabilities, randomSource = random):
        # List of random numbers
        self._random_nums = random_nums 
        # List of probabilities
        self._probabilities = probabilities
        self.randomSource = randomSource

        
        # check number of elements in both lists are the same       
        if len(self._random_nums) != len(self._probabilities):
            raise ValueError('The length of random numbers list should be equal to the length of probabilities list.')

        # check the sum of probabilities==1
        if not math.isclose(sum(self._probabilities), 1, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError('Sum of probabilities should be eqaul to 1.')
   
    def next_num(self):        
        # List of probability region for each number in list
        # How to make this O(logn) time constant?
        probability_regions = list(accumulate(self._probabilities))

        

        # Hash table of (probability_regions, random_nums) as key-value pairs
        values=self._random_nums
        keys=probability_regions
        hash_table = dict(zip(keys, values))

        # Random probability
        rand_prob = self.randomSource.random()
                
        # Binary search to find the upper bound for generated random probability 
        upper_bound = self._binarySearch(probability_regions, rand_prob)
                       
        return hash_table[upper_bound]
    
    def _binarySearch(self, alist, item):
        # if alist empty, return
        if not alist:
            return
        # Middle element index
        mid=len(alist)//2
        # Return list value for one element list
        if len(alist)<=1:
            return alist[0]
        # Once item range is found, return the upper bound
        elif (alist[mid]>=item) and (alist[mid-1]<item):
            return alist[mid]
        # Recursion search
        elif alist[mid] > item:
            return self._binarySearch(alist[:mid], item)
        else:
            return self._binarySearch(alist[mid+1:], item)  

class ConstRandom(object):
    def __init__(self, num):
        self.num = num

    def random(self):
        return self.num




import unittest

class RandomGenTest(unittest.TestCase):

    def testInputLength(self):
        random_nums = [2,5]
        probabilities = [0.1,0.4, 0.5]
        self.assertRaises(Exception, RandomGen, random_nums, probabilities) 

# mock unittest from python
    def testProbabilitiesSum(self):
        probabilities = [0.3,0.5]
        random_nums = [5,10]
        self.assertRaises(Exception, RandomGen, random_nums, probabilities) 


    def testExactInput(self):
        random_nums = [2,5, 10]
        probabilities = [0.1,0.4, 0.5]
        obj = RandomGen(random_nums, probabilities, ConstRandom(0.1)) 
        self.assertEqual(obj.next_num(), 2)
        #self.assertRaises(Exception, RandomGen, random_nums, probabilities, True, 0.1) 


if __name__ == '__main__':
    # Test case for 1000 times
    random_nums, probabilities = [-1,0,1,2,3], [0.01, 0.3, 0.58, 0.1, 0.01]
    obj = RandomGen(random_nums, probabilities)
    n=10000
    results = dict(zip(random_nums, [0]*len(random_nums)))

    for i in range(n):
        num = obj.next_num()
        results[num]+=1/n

    print(results)

    # Run test units
    unittest.main()

