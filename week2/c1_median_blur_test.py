import os
import sys
from c1_median_blur import *
import numpy as np
import cv2


def test_replicaPaddingIndex():
    row = 4
    col = 5
    img = np.arange(0, row * col).reshape(row, col)
    assert(replicaPaddingIndex(img, -1, -1) == 0)
    assert(replicaPaddingIndex(img, 2, -1) == 10)
    assert(replicaPaddingIndex(img, 2, col) == 14)
    assert(replicaPaddingIndex(img, 2, col+1) == 14)
    assert(replicaPaddingIndex(img, -1, 2) == 2)
    assert(replicaPaddingIndex(img, row, 2) == 17)
    assert(replicaPaddingIndex(img, row+1, 2) == 17)


def test_zeroPaddingIndex():
    row = 4
    col = 5
    img = np.arange(0, row * col).reshape(row, col)
    assert(zeroPaddingIndex(img, -1, -1) == 0)
    assert(zeroPaddingIndex(img, 2, -1) == 0)
    assert(zeroPaddingIndex(img, 2, col) == 0)
    assert(zeroPaddingIndex(img, 2, col+1) == 0)
    assert(zeroPaddingIndex(img, -1, 2) == 0)
    assert(zeroPaddingIndex(img, row, 2) == 0)
    assert(zeroPaddingIndex(img, row+1, 2) == 0)


def test_medianBlur():
    img = np.full((4, 5), 5, dtype='uint8')
    kernel = np.zeros((3, 3))

    assert((medianBlur(img, kernel, 'REPLICA') == img).all())

    img = np.full((4, 5), 0, dtype='uint8')
    assert((medianBlur(img, kernel, 'ZERO') == img).all())


def test_quickSelectMedianValue():
    nums = np.random.randint(0, 256, (1000,), dtype='uint8')
    medianValue = quickSelectMedianValue(nums.copy())
    nums.sort()
    assert(medianValue == nums[(len(nums)-1)//2])


def test_medianBlurQuickSelect():
    img = np.random.randint(0, 256, (512, 512), dtype='uint8')
    kernel = np.zeros((4, 4), dtype='uint8')
    filted1 = medianBlur(img, kernel, 'REPLICA')
    filted2 = medianBlurQuickSelect(img, kernel, 'REPLICA')

    assert((filted1 == filted2).all())


def test_quickSort():
    nums = np.random.randint(0, 256, (1000,), dtype='uint8')

    nums_copy = nums.copy()
    nums_copy.sort()

    quickSort(nums)
    assert((nums == nums_copy).all())


def test():
    test_replicaPaddingIndex()
    test_zeroPaddingIndex()
    test_medianBlur()

    test_quickSelectMedianValue()
    test_medianBlurQuickSelect()

    test_quickSort()


if __name__ == "__main__":
    # test()

    filepath = sys.argv[1]
    if not filepath:
        proj_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(os.path.dirname(proj_dir), 'assets')
        filepath = os.path.join(assets_dir, 'lenna.jpg')

    img = cv2.imread(filepath, 0)
    kernel = np.zeros((5, 5))

    print("img: {}, kernel: {}".format(img.shape, kernel.shape))

    cv2.imshow('lenna', img)

    blured = medianBlur(img, kernel, 'REPLICA')
    cv2.imshow('medianBlur', blured)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
