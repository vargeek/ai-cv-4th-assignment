#include "c1_median_blur.hpp"
#include <algorithm>

static u8 replicaPaddingIndex(vector<vector<u8>> &img, int row, int col);
static u8 zeroPaddingIndex(vector<vector<u8>> &img, int row, int col);

vector<vector<u8>> medianBlur(vector<vector<u8>> &img, vector<vector<u8>> kernel, string padding_way) {
    std::transform(padding_way.begin(), padding_way.end(), padding_way.begin(), ::toupper);

    u8 (*index_fn)(vector<vector<u8>> &, int, int) = padding_way == "REPLICA" ? replicaPaddingIndex : zeroPaddingIndex;

    int rows = img.size();
    int cols = rows > 0 ? img[0].size() : 0;
    int krows = kernel.size();
    int kcols = krows > 0 ? kernel[0].size() : 0;
    int ksize = krows * kcols;

    int row_offset = -(krows / 2);
    int col_offset = -(kcols / 2);

    vector<u8> pixels = vector<u8>(ksize);
    vector<vector<u8>> result = vector<vector<u8>>(rows);
    for (int row = 0; row < rows; row++) {
        result[row] = vector<u8>(cols);
    }

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            int row_start = row + row_offset;
            int col_start = col + col_offset;
            int idx = 0;
            for (int r = row_start; r < row_start + krows; r++) {
                for (int c = col_start; c < col_start + kcols; c++) {
                    pixels[idx++] = index_fn(img, r, c);
                }
            }
            sort(pixels.begin(), pixels.end());
            result[row][col] = pixels[(ksize - 1) / 2];
        }
    }

    return result;
}

vector<vector<u8>> medianBlurQuickSort(vector<vector<u8>> &img, vector<vector<u8>> kernel, string padding_way) {
    std::transform(padding_way.begin(), padding_way.end(), padding_way.begin(), ::toupper);

    u8 (*index_fn)(vector<vector<u8>> &, int, int) = padding_way == "REPLICA" ? replicaPaddingIndex : zeroPaddingIndex;

    int rows = img.size();
    int cols = rows > 0 ? img[0].size() : 0;
    int krows = kernel.size();
    int kcols = krows > 0 ? kernel[0].size() : 0;
    int ksize = krows * kcols;

    int row_offset = -(krows / 2);
    int col_offset = -(kcols / 2);

    vector<u8> pixels = vector<u8>(ksize);
    vector<vector<u8>> result = vector<vector<u8>>(rows);
    for (int row = 0; row < rows; row++) {
        result[row] = vector<u8>(cols);
    }

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            int row_start = row + row_offset;
            int col_start = col + col_offset;
            int idx = 0;
            for (int r = row_start; r < row_start + krows; r++) {
                for (int c = col_start; c < col_start + kcols; c++) {
                    pixels[idx++] = index_fn(img, r, c);
                }
            }
            quickSort(pixels);
            result[row][col] = pixels[(ksize - 1) / 2];
        }
    }

    return result;
}

vector<vector<u8>> medianBlurQuickSelect(vector<vector<u8>> &img, vector<vector<u8>> kernel, string padding_way) {
    std::transform(padding_way.begin(), padding_way.end(), padding_way.begin(), ::toupper);

    u8 (*index_fn)(vector<vector<u8>> &, int, int) = padding_way == "REPLICA" ? replicaPaddingIndex : zeroPaddingIndex;

    int rows = img.size();
    int cols = rows > 0 ? img[0].size() : 0;
    int krows = kernel.size();
    int kcols = krows > 0 ? kernel[0].size() : 0;
    int ksize = krows * kcols;

    int row_offset = -(krows / 2);
    int col_offset = -(kcols / 2);

    vector<u8> pixels = vector<u8>(ksize);
    vector<vector<u8>> result = vector<vector<u8>>(rows);
    for (int row = 0; row < rows; row++) {
        result[row] = vector<u8>(cols);
    }

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            int row_start = row + row_offset;
            int col_start = col + col_offset;
            int idx = 0;
            for (int r = row_start; r < row_start + krows; r++) {
                for (int c = col_start; c < col_start + kcols; c++) {
                    pixels[idx++] = index_fn(img, r, c);
                }
            }
            result[row][col] = quickSelectMedianValue(pixels);
        }
    }

    return result;
}
static u8 replicaPaddingIndex(vector<vector<u8>> &img, int row, int col) {
    int rows = img.size();
    int cols = rows > 0 ? img[0].size() : 0;
    row = max<u8>(0, min<u8>(row, rows - 1));
    col = max<u8>(0, min<u8>(col, cols - 1));
    return img[row][col];
}

static u8 zeroPaddingIndex(vector<vector<u8>> &img, int row, int col) {
    int rows = img.size();
    int cols = rows > 0 ? img[0].size() : 0;
    bool inside = row >= 0 && row < rows && col >= 0 && col < cols;

    return inside ? img[row][col] : 0;
}

// vector<vector<int>>& img, vector<vector<int>> kernel, string padding_way

static void _quickSort(vector<u8> &nums, int l, int r) {
    int i = l, j = r, val = nums[(l + r) / 2];
    while (i <= j) {
        while (i <= j && nums[i] < val) {
            i++;
        }
        while (i <= j && nums[j] > val) {
            j--;
        }
        if (i <= j) {
            u8 tmp = nums[i];
            nums[i] = nums[j];
            nums[j] = tmp;
            i++;
            j--;
        }
    }
    if (l <= j) {
        _quickSort(nums, l, j);
    }
    if (i <= r) {
        _quickSort(nums, i, r);
    }
}
void quickSort(vector<u8> &nums) {
    _quickSort(nums, 0, nums.size() - 1);
}

static u8 _kthSmallest(vector<u8> &nums, int k, int l, int r) {
    if (r <= l) {
        return nums[k];
    }

    int i = l, j = r, val = nums[(l + r) / 2];
    while (i <= j) {
        while (i <= j && nums[i] < val) {
            i++;
        }
        while (i <= j && nums[j] > val) {
            j--;
        }
        if (i <= j) {
            u8 tmp = nums[i];
            nums[i] = nums[j];
            nums[j] = tmp;
            i++;
            j--;
        }
    }
    if (k <= j) {
        return _kthSmallest(nums, k, l, j);
    }
    if (k >= i) {
        return _kthSmallest(nums, k, i, r);
    }
    return nums[k];
}

u8 quickSelectMedianValue(vector<u8> &nums) {
    int r = nums.size() - 1;
    return _kthSmallest(nums, r / 2, 0, r);
}
