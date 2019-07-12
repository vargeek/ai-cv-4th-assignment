import numpy as np
import os
import sys
import cv2


def medianBlur(img, kernel, padding_way):
    """
    img & kernel is List of List  
    padding_way a string
    """

    index_fn = padding_way.upper() == 'REPLICA' and replicaPaddingIndex or zeroPaddingIndex
    rows, cols, *_ = img.shape
    krows, kcols, *_ = kernel.shape
    ksize = krows * kcols
    row_offset = -(krows // 2)
    col_offset = -(kcols // 2)

    pixels = np.zeros((ksize,), dtype=img.dtype)
    result = np.zeros((rows, cols), dtype=img.dtype)

    for row in range(0, rows):
        for col in range(0, cols):
            row_start = row + row_offset
            col_start = col + col_offset
            idx = 0
            for r in range(row_start, row_start + krows):
                for c in range(col_start, col_start + kcols):
                    pixels[idx] = index_fn(img, r, c)
                    idx = idx + 1
            pixels.sort()
            # quickSort(pixels)
            result[row][col] = pixels[(ksize-1)//2]
    return result


def medianBlurQuickSelect(img, kernel, padding_way):
    """
    medianBlur(使用QuickSelect算法查找中位数)  
    img & kernel is List of List  
    padding_way a string
    """

    index_fn = padding_way.upper() == 'REPLICA' and replicaPaddingIndex or zeroPaddingIndex
    rows, cols, *_ = img.shape
    krows, kcols, *_ = kernel.shape
    ksize = krows * kcols
    row_offset = -(krows // 2)
    col_offset = -(kcols // 2)

    pixels = np.zeros((ksize,), dtype=img.dtype)
    result = np.zeros((rows, cols), dtype=img.dtype)

    for row in range(0, rows):
        for col in range(0, cols):
            row_start = row + row_offset
            col_start = col + col_offset
            idx = 0
            for r in range(row_start, row_start + krows):
                for c in range(col_start, col_start + kcols):
                    pixels[idx] = index_fn(img, r, c)
                    idx = idx + 1

            result[row][col] = quickSelectMedianValue(pixels)
    return result


def replicaPaddingIndex(img, row, col):
    """
    REPLICA padding 索引方式
    """
    shape = img.shape
    row = max(0, min(row, shape[0]-1))
    col = max(0, min(col, shape[1]-1))
    return img[row, col]


def zeroPaddingIndex(img, row, col):
    """
    ZERO padding 索引方式
    """
    shape = img.shape
    inside = row >= 0 and row < shape[0] and col >= 0 and col < shape[1]
    return inside and img[row, col] or 0


def _kthSmallest(nums, k, l, r):
    if r <= l:
        return nums[k]
    i, j, val = l, r, nums[(l+r)//2]
    while i <= j:
        while i <= j and nums[i] < val:
            i = i + 1
        while i <= j and nums[j] > val:
            j = j - 1
        if i <= j:
            nums[i], nums[j] = nums[j], nums[i]
            i = i + 1
            j = j - 1

    if k <= j:
        return _kthSmallest(nums, k, l, j)
    if k >= i:
        return _kthSmallest(nums, k, i, r)
    return nums[k]


def quickSelectMedianValue(nums):
    r = len(nums) - 1
    return _kthSmallest(nums, r//2, 0, r)


def _quickSort(nums, l, r):
    i, j, val = l, r, nums[(l+r)//2]
    while i <= j:
        while i <= j and nums[i] < val:
            i = i + 1
        while i <= j and nums[j] > val:
            j = j - 1
        if i <= j:
            nums[i], nums[j] = nums[j], nums[i]
            i = i + 1
            j = j - 1
    if l <= j:
        _quickSort(nums, l, j)
    if i <= r:
        _quickSort(nums, i, r)


def quickSort(nums):
    _quickSort(nums, 0, len(nums)-1)


if __name__ == "__main__":
    filepath = len(sys.argv) > 1 and sys.argv[1] or None
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
