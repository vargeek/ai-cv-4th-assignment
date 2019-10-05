# week10

## Coding

NMS: According to what we are discussing about NMS, please write your own code for NMS in C++ or Python. For c++, please finish the function:

```c++
vector<vector<float>> NMS(vector<vector<float>> lists, float thre)
{
// lists[0:4]: x1, x2, y1, y2; lists[4]: score
}
```

For Python, please finish the function:

```python
def NMS(lists, thre):
# lists is a list. lists[0:4]: x1, x2, y1, y2; lists[4]: score
```

- [x] c++: [NMS.cpp](./NMS.cpp)  

  ```shell
  cd cmake && cmake . && make week10_test && ./build/week10_test
  ```

- [x] python: [nms.py](./nms.py)  

  ```shell
  python -m nms_test
  ```
