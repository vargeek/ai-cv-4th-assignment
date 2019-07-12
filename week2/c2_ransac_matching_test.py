import unittest
import numpy as np
import cv2
import c2_ransac_matching as c2


class Test(unittest.TestCase):
    def test_checkPoints(self):
        points = [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
        ]
        self.assertTrue(c2.checkPoints(points))

        points = [
            [0, 0],
            [1, 0],
            [2, 0],
            [0, 1],
        ]
        self.assertFalse(c2.checkPoints(points))

        points = [
            [0, 0],
            [1, 0.1],
            [2, 0],
            [0, 1],
        ]
        self.assertTrue(c2.checkPoints(points))

    def test_randomPoints(self):
        A = [
            [285, 281],
            [139, 51],
            [320, 281],
            [263, 362],
            [323, 359],
            [390, 511],
            [380, 479],
            [352, 139],
            [376, 211],
            [258, 242],
            [339, 200],
            [106, 212],
            [371, 192],
            [78, 272],
            [30, 511],
        ]
        B = [
            [284, 259],
            [103, 180],
            [304, 245],
            [303, 317],
            [339, 289],
            [439, 353],
            [419, 337],
            [266, 146],
            [310, 181],
            [252, 247],
            [283, 189],
            [148, 290],
            [299, 170],
            [157, 337],
            [223, 498],
        ]
        (a, b, indices) = c2.randomPoints(A, B)
        self.assertTrue(len(a) == 4)
        self.assertTrue(len(b) == 4)
        self.assertEqual(len(a), len(b))
        for (i, idx) in enumerate(indices):
            self.assertEqual(a[i], A[idx])
            self.assertEqual(b[i], B[idx])
            self.assertTrue(c2.checkPoints(a))
            self.assertTrue(c2.checkPoints(b))


if __name__ == "__main__":
    unittest.main()
