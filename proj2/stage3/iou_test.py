import unittest
from math import fabs
from iou import IOU

kDelta = 0.00001

class Test(unittest.TestCase):
    def test_iou(self):

        self.assertTrue(fabs(IOU([0,0,1,1], [1,1,2,2]) - 0) < kDelta)

        self.assertTrue(fabs(IOU([0,0,1,1], [0.5,0.5,1.5,1.5]) - 1/7.0) < kDelta)

        self.assertTrue(fabs(IOU([0,0,1,1], [0,0,1,1]) - 1) < kDelta)

        self.assertTrue(fabs(IOU([0,0,1,1], [0.1,0,1.1,1]) - 0.9/1.1) < kDelta)

if __name__ == "__main__":
    unittest.main()