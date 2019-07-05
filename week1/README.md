# week1

- [x] 1. Recode all examples;

  [q1_recode.py](./q1_recode.py)

  执行 `python q1_recode.py` 查看运行效果。选中窗口后点击任意键查看下一个窗口。
---

- [x] 2. Please combine image crop, color shift, rotation and perspective transform together to complete a data augmentation script.  
  + Your code need to be completed in Python/C++ in .py or .cpp file with comments and readme file to indicate how to use.  
 
  [q2_augmentation.py](./q2_augmentation.py)

  执行 `python q2_augmentation.py` 查看处理效果，默认对 [assets/lenna.jpg](../assets/lenna.jpg) 图片进行五次随机处理。  
  也可以在命令后面传入多个图片文件地址以查看对任意图片的处理：
    ```
    python q2_augmentation.py path/to/file1 path/to/file2 ...
    ```

  源文件中定义的 `crop_image`, `light_color`, `gamma_correction`, `equalize_y_hist`, `rotate_image`, `perspective_transform_image` 等函数分别对图片进行“裁剪”、“亮化”、“gamma校正”等处理。  
  `random_augmentation` 函数将这几个处理组合起来。

---

- [ ] 3. Do think about your own interests very carefully and choose your topic within 3 weeks.
---

- [ ] 4. 【Please send answers to those questions to mqgao@kaikeba.com】  
  Please answer some questions about our course. We do appreciate your help.  
  4.1 What do you want to get in this course?  
  4.2 What problems do you want to solve?  
  4.3 What advantages do you have to accomplish your goal?  
  4.4 What disadvantages you need to overcome to accomplish your goal?  
  4.5 How will you plan to study in this course?
