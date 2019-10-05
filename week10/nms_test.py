import unittest
from math import fabs
from nms import IOU,NMS

kDelta = 0.00001

class Test(unittest.TestCase):
    def test_iou(self):

        self.assertTrue(fabs(IOU([0,1,0,1], [1,2,1,2]) - 0) < kDelta)

        self.assertTrue(fabs(IOU([0,1,0,1], [0.5,1.5,0.5,1.5]) - 1/7.0) < kDelta)

        self.assertTrue(fabs(IOU([0,1,0,1], [0,1,0,1]) - 1) < kDelta)

        self.assertTrue(fabs(IOU([0,1,0,1], [0.1,1.1,0,1]) - 0.9/1.1) < kDelta)

    def test_nms(self):
        input = [
            [1, 4, 1, 4, 9],
            [-3+0.1, -1+0.1, -3+0.1, -1+0.1, 7],
            [40, 43, 40, 43, 12],
            [0, 3, 0, 3, 10],
            [10, 12, 10, 12, 6],
            [-2+0.1, 0.1, -2+0.1, 0.1, 8],
            [20, 22, 20, 22, 3],
            [11, 13, 11, 13, 5],
            [41, 42, 41, 42, 11],
            [32-0.1, 34-0.1, 32-0.1, 34-0.1, 1],
            [30, 32, 30, 32, 2],
            [20, 22, 20, 22, 4],
        ]
        threshold = 0.1
        
        results = NMS(input, threshold)
        got = [int(r[4]) for r in results]
        want = [12, 10, 8, 6, 4, 2, 1]
        self.assertListEqual(got, want)
        

if __name__ == "__main__":
    unittest.main()
