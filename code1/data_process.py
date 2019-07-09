from config import Config
import os
import numpy as np
from scipy import misc

def get_image_paths(facedir):
    '''
    :param facedir:     不同人物对应的图像文件夹
    :return:             文件夹下的所有图像的路径
    '''
    image_paths = []
    # 如果是一个文件夹 则读取
    if os.path.isdir(facedir):
        image_paths = [os.path.join(facedir,img) for img in os.listdir(facedir)]
    return image_paths

def get_data_set(data_dir):
    '''
    :param data_dir:  总的数据集目录
    :return:          数据集下的 人物id及对应的图片路径字典
    '''
    dataset = {}
    # 读出所有的类别
    classes = [path for path in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, path))]
    classes.sort()

    # 总的类别数
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        face_dir = os.path.join(data_dir, class_name)
        images_paths = get_image_paths(face_dir)

        dataset[class_name] = images_paths

    return dataset

#从数据集中得到图片和标签片
def get_image_paths_and_labels(dataset):
    '''
    :param dataset:     即上面函数返回的字典格式的人物id与对应图像路径
    :return:            两个对应的列表，一个是图片路径，一个是label
    '''
    image_paths_flat, labels_flat = [], []

    for (idx, value) in enumerate(dataset.values()):
        image_paths_flat += value
        labels_flat += [idx] * len(value)

    return image_paths_flat, labels_flat

#随机旋转图片
def random_rotate_image(image):
    angle = np.random.uniform(low=-10.0,high=10.0)
    return misc.imrotate(image,angle,'bicubic')

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std,1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x,mean),1/std_adj)
    return y