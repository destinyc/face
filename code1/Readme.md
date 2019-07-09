# 程序运行说明

---

## 1. 文件说明

face_verification/
|-----data  数据集目录
|-----code  代码目录
      |-------------utils.py                    一些辅助函数
      |-------------data_process.py             一些数据处理函数
      |-------------inception_resnet_v1.py      网络框架函数
      |-------------train_center_softmax_loss.py训练函数
      |-------------eval.py                     训练结果评估函数
      |-------------compare.py                  课程要求的接口函数
      |-------------model                       训练好的权重文件夹


## 2. 程序运行需要的平台
- python 3.6.5
- numpy 1.15.1
- tensorflow 1.9.0
- cv2 3.4.2

## 3.程序运行说明

-----------train_center_softmax_loss.py直接运行可以对网络进行训练（需要在data目录下放置数据集）在config文件下有一个变量pretrained_model可以选择预训练模型

-----------eval.py直接运行可以得到训练结果的评估（同样需要数据集），在config文件下变量test_number可以控制测试正负对儿的数量。结果会输出最佳阈值和对应准确率以及阈值相关的ROC曲线。

-----------compare.py比较两张图片是否是同一个人，需要输入两个变量，终端运行命令如下：
			
			python compare.py 图片1路径 图片2路径      （用绝对路径会保险一点）