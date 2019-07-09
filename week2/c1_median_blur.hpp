#if !defined(__C1_MEDIAN_BLUR_HPP__)
#define __C1_MEDIAN_BLUR_HPP__

#include <string>
#include <vector>
using namespace std;
typedef unsigned char u8;

vector<vector<u8>> medianBlur(vector<vector<u8>> &img, vector<vector<u8>> kernel, string padding_way);
vector<vector<u8>> medianBlurQuickSort(vector<vector<u8>> &img, vector<vector<u8>> kernel, string padding_way);
vector<vector<u8>> medianBlurQuickSelect(vector<vector<u8>> &img, vector<vector<u8>> kernel, string padding_way);

void quickSort(vector<u8> &nums);

u8 quickSelectMedianValue(vector<u8> &nums);

#endif  // __C1_MEDIAN_BLUR_HPP__
