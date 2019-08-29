# p2

紫色:要学习的内容
红色:项目任务(或注意事项)
绿色:要回答的问题

21点
可能有<0 的情况出现，大家可以当做 0 或 者忽略该行信息即可

可以选取原 始人脸框的 0.25 倍进行 expand。expand 时，请注意扩增后的人脸框不要超过图像 大小。

使用空白图片得到的loss也是20+?

loss versus num_samples

数据：
不要 normalize 可以么
水平翻转、小角度旋转、是否可以平移

训练方法：
Adam， lr
如果一开始用 Adam，之后换成 sgd 呢
step 改变 lr
batch normalization

网络：
resnet
fpn
其他的
过拟

目标：
回归 heatmap

loss：
smooth l1 loss
