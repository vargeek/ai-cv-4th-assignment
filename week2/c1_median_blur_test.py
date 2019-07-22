import c1_median_blur as c1
import numpy as np
import unittest


class Test(unittest.TestCase):
    def test_replicaPaddingIndex(self):
        row = 4
        col = 5
        img = np.arange(0, row * col).reshape(row, col)
        self.assertEqual(c1.replicaPaddingIndex(img, -1, -1), 0)
        self.assertEqual(c1.replicaPaddingIndex(img, 2, -1), 10)
        self.assertEqual(c1.replicaPaddingIndex(img, 2, col), 14)
        self.assertEqual(c1.replicaPaddingIndex(img, 2, col+1), 14)
        self.assertEqual(c1.replicaPaddingIndex(img, -1, 2), 2)
        self.assertEqual(c1.replicaPaddingIndex(img, row, 2), 17)
        self.assertEqual(c1.replicaPaddingIndex(img, row+1, 2), 17)

    def test_zeroPaddingIndex(self):
        row = 4
        col = 5
        img = np.arange(0, row * col).reshape(row, col)
        self.assertEqual(c1.zeroPaddingIndex(img, -1, -1), 0)
        self.assertEqual(c1.zeroPaddingIndex(img, 2, -1), 0)
        self.assertEqual(c1.zeroPaddingIndex(img, 2, col), 0)
        self.assertEqual(c1.zeroPaddingIndex(img, 2, col+1), 0)
        self.assertEqual(c1.zeroPaddingIndex(img, -1, 2), 0)
        self.assertEqual(c1.zeroPaddingIndex(img, row, 2), 0)
        self.assertEqual(c1.zeroPaddingIndex(img, row+1, 2), 0)

    def test_medianBlur(self):
        kernel = np.zeros((3, 3))
        img = np.array([
            [217, 159, 25, 51, 42],
            [46, 216, 229, 91, 90],
            [64, 69, 140, 56, 20],
            [197, 102, 180, 94, 94]], dtype='uint8')
        expected = np.array([
            [216, 159, 91, 51, 51],
            [69, 140, 91, 56, 51],
            [69, 140, 102, 94, 90],
            [102, 140, 102, 94, 94]], dtype='uint8')
        self.assertTrue(
            (c1.medianBlur(img, kernel, 'REPLICA') == expected).all())

        img = np.array([
            [14, 42, 153, 230, 87],
            [36, 67, 138, 26, 118],
            [151, 158, 18, 58, 94],
            [133, 107, 140, 107, 201]], dtype='uint8')
        expected = np.array([
            [0, 36, 42, 87, 0],
            [36, 67, 67, 94, 58],
            [67, 133, 107, 107, 58],
            [0, 107, 58, 58, 0]], dtype='uint8')
        self.assertTrue((c1.medianBlur(img, kernel, 'ZERO') == expected).all())

    def test_quickSelectMedianValue(self):
        nums = np.random.randint(0, 256, (1000,), dtype='uint8')
        medianValue = c1.quickSelectMedianValue(nums.copy())
        nums.sort()
        self.assertTrue(medianValue == nums[(len(nums)-1)//2])

    def test_medianBlurQuickSelect(self):
        img = np.random.randint(0, 256, (512, 512), dtype='uint8')
        kernel = np.zeros((4, 4), dtype='uint8')
        filted1 = c1.medianBlur(img, kernel, 'REPLICA')
        filted2 = c1.medianBlurQuickSelect(img, kernel, 'REPLICA')

        self.assertTrue((filted1 == filted2).all())

    def test_quickSort(self):
        nums = np.random.randint(0, 256, (1000,), dtype='uint8')

        nums_copy = nums.copy()
        nums_copy.sort()

        c1.quickSort(nums)
        self.assertTrue((nums == nums_copy).all())

    def test_medianBlurHistogram(self):
        img = np.random.randint(0, 256, (512, 512), dtype='uint8')
        kernel = np.zeros((4, 4), dtype='uint8')
        filted1 = c1.medianBlurQuickSelect(img, kernel, 'REPLICA')
        filted2 = c1.medianBlurHistogram(img, kernel, 'REPLICA')

        self.assertTrue((filted1 == filted2).all())


if __name__ == "__main__":
    unittest.main()
