from unittest import TestCase
from randomNumber import RandomGenTest

class RandomGenTest(TestCase):

    # def setUp(self):
        

    def test_lens(self):
        self.assertCountEqual(ValueError('The length of random numbers list should be equal to the length of probabilities list.'), RandomGenTest([1,2],[0.1,0.5,0.4]))
    
    

if __name__ == '__main__':
    unittest.main()