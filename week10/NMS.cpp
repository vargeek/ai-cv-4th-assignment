#include "NMS.hpp"
#include <algorithm>

inline float area(const vector<float> &roi) {
    return (roi[1] - roi[0]) * (roi[3] - roi[2]);
}

/**
 * @brief intersection over union
 * 
 * @param roi1 [x1,x2,y1,y2]
 * @param roi2 [x1,x2,y1,y2]
 * @return float 
 */
float IOU(const vector<float> &roi1, const vector<float> &roi2) {
    const vector<float> &left = roi1[0] < roi2[0] ? roi1 : roi2;
    const vector<float> &right = roi1[0] < roi2[0] ? roi2 : roi1;
    const vector<float> &top = roi1[2] < roi2[2] ? roi1 : roi2;
    const vector<float> &bottom = roi1[2] < roi2[2] ? roi2 : roi1;

    if (right[0] > left[1] || bottom[2] > top[3])
    {
        return 0;
    }

    float w = left[1] - right[0];
    float h = top[3] - bottom[2];

    float intersection_area = w * h;
    float union_area = area(roi1) + area(roi2) - intersection_area;
    
    return intersection_area / union_area;
}

/**
 * @brief Non-Maximum Suppression
 * 
 * @param lists lists[0:4]: x1, x2, y1, y2; lists[4]: score
 * @param threshold 
 * @return vector<vector<float>> 
 */
vector<vector<float>> NMS(vector<vector<float>> lists, float threshold) {
    vector<vector<float>> results;

    sort(lists.begin(), lists.end(), [](vector<float> &x, vector<float> &y){
        return y[4] < x[4];
    });

    size_t i = 0;
    while (i < lists.size())
    {
        vector<float> &tmp = lists[i];
        results.push_back(lists[i++]);

        for (size_t j = i; j < lists.size(); j++)
        {
            if (IOU(tmp, lists[j]) > threshold)
            {
                swap(lists[i], lists[i]);
                i++;
            }
        }
    }

    return results;
}
