#if !defined(__NMS_HPP__)
#define __NMS_HPP__
#include <vector>

using namespace std;

float IOU(const vector<float> &roi1, const vector<float> &roi2);

vector<vector<float>>  NMS(vector<vector<float>> lists, float threshold);

#endif  // __NMS_HPP__
