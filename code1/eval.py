import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
import utils
import data_process
import cv2
import random
from config import Config
import matplotlib.pyplot as plt
import os

'''
            # 准确率  acc = 正确预测的样本数 / 总样本数
            
            
            # 精准率  precision = TP / (TP + FP)    正确预测的  正例  占 预测结果中正例 的比例
            # 召回率  recall =    TP / (TP + FN)    正确预测的  正例  占 实际标签是正例 的比例
            
            # ROC曲线     横坐标是 FPR = FP / (FP + TN)
            #             纵坐标是 TPR = TP / (TP + FN)  也是召回率
            
            # AUC  : ROC曲线的积分面积
            
            
'''

def calculate_accuracy(threshold, dist, actual_issame):
    '''
    :param threshold:
    :param dist:                  这里是测试数据对对应的特征距离
    :param actual_issame:
    :return:
    '''
    predict = np.less(dist, threshold)



    # 计算 F1 score 用到的四个参数
    tp = np.sum(np.logical_and(predict, actual_issame))          # 真正数量
    fp = np.sum(np.logical_and(predict, np.logical_not(actual_issame)))      # 本应为假预测成真
    tn = np.sum(np.logical_and(np.logical_not(predict), np.logical_not(actual_issame)))   # 真负数量
    fn = np.sum(np.logical_and(np.logical_not(predict), actual_issame))      #本应为真预测成假

    # 预测的正确的正例所占总正例的比例， 等价于召回率
    tpr = 0 if (tp+fn==0) else float(tp)/float(tp+fn)
    # 预测的正确的反例所占总反例的比例
    fpr = 0 if (fp+tn==0) else float(fp)/float(fp+tn)                  # 这两个指标是用于计算 ROC曲线的

    acc = float(tp + tn) / dist.size

    return tpr, fpr, acc


def calculate_ROC(thresholds, embeddings1, embeddings2, actual_issame):
    '''
    TPR = TP / (TP + FN)      FPR = FP / (FP + TN)
    :param thresholds:          这里输入的阈值是一个列表，用来计算随着阈值的变化，fpr、tpr、acc的变化
    :param embeddings1:
    :param embeddings2:          整个测试集对儿的特征向量
    :param actual_issame:
    :param nrof_folds:
    :return:                    40维的tpr（纵坐标）和fpr（横坐标）用于画ROC，加上10维的准确率(最佳阈值下的10次交叉验证)
    '''
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])

    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)

    tprs = np.zeros((nrof_thresholds))
    fprs = np.zeros((nrof_thresholds))

    dist = np.sum(np.square(np.subtract(embeddings1, embeddings2)), axis=1)
    indices = np.arange(nrof_pairs)
    best_threshold = 0.0
    best_accuracy = 0

    # 得到每次交叉验证下不同阈值对应的 tpr、fpr
    for threshold_idx, threshold in enumerate(thresholds):
        tprs[threshold_idx], fprs[threshold_idx], accuracy = calculate_accuracy(threshold, dist, actual_issame)
        if accuracy > best_accuracy:
            best_threshold = threshold
            best_accuracy = accuracy


    return tprs, fprs, best_threshold, best_accuracy


def load_data(image_paths, image_size):

    img_list = []
    print(image_paths[0] + ' ' + image_paths[1])
    for image in image_paths:
        img = cv2.imread(image)
        img = cv2.resize(img, (image_size, image_size))

        prewhitened = data_process.prewhiten(img)
        img_list.append(prewhitened)

    images = np.stack(img_list)

    return images


def get_eval_data():
    '''

    :param data_set:   之前处理过的字典格式的数据
    :return:           采样出来的样本对列表和对应的是否为同一个人的label列表
    '''

    data_set = data_process.get_data_set(Config.data_dir)
    images_couple, labels = [], []

    # 采样正样本对
    test_same_label = random.sample(list(data_set.keys()), Config.test_number)
    for label in test_same_label:
        image1, image2 = random.sample(list(data_set[label]),2)

        images_couple.append((image1, image2))
        labels.append(1)

    # 采样负样本对
    anchor_label = random.sample(list(data_set.keys()), 1)
    anchor_image = random.sample(list(data_set[anchor_label[0]]),1)

    label_list = list(data_set.keys())
    label_list.remove(anchor_label[0])
    not_same_labels = random.sample(label_list, Config.test_number)

    for neg_label in not_same_labels:
        neg_image = random.sample(list(data_set[neg_label]),1)
        images_couple.append((anchor_image[0], neg_image[0]))
        labels.append(0)



    return images_couple, labels

def draw_ROC():
    images_couple, labels = get_eval_data()
    print(len(images_couple), len(labels))

    randnum = random.randint(0, 100)
    random.seed(randnum)
    random.shuffle(images_couple)
    random.seed(randnum)
    random.shuffle(labels)

    embs1, embs2 = [], []
    with tf.Graph().as_default():
        with tf.Session() as sess:
            #加载模型
            utils.load_model(Config.model)
            images_placeholder = tf.get_default_graph().get_tensor_by_name('batch_join:0')
            embeddings = tf.get_default_graph().get_tensor_by_name('embeddings:0')
            train_flag = tf.get_default_graph().get_tensor_by_name('phase_train:0')

            for idx, images in enumerate(images_couple):
                images = load_data(images, Config.image_size)
                feed_dict = {images_placeholder:images,train_flag:False}
                emb = sess.run(embeddings,feed_dict=feed_dict)
                embs1.append(emb[0])
                embs2.append(emb[1])

    embs1 = np.asarray(embs1)
    embs2 = np.asarray(embs2)
    thresholds_list = np.linspace(0, 4.0, 40)

    tpr, fpr, best_threshlod, acc = calculate_ROC(thresholds_list, embs1, embs2, labels)
    print(tpr, fpr)
    print('best_threshold : ', best_threshlod)
    print('best_accuracy: ',acc)

    plt.plot(fpr, tpr, 'r-')
    plt.title('ROC curve')
    plt.show()


def read_data():
    pairs = []
    with open('..\\data\\data\\pairs1_nolabel.txt', 'r') as f:
        for line in f:
            pairs.append([os.path.join('..\\data\\data\\images_aligned_2018Autumn',image) for image in line.strip().split('  ')])

    return pairs


def test():
    images_couple= read_data()
    print(len(images_couple))
    labels = []

    with tf.Graph().as_default():
        with tf.Session() as sess:
            #加载模型
            utils.load_model(Config.model)
            images_placeholder = tf.get_default_graph().get_tensor_by_name('batch_join:0')
            embeddings = tf.get_default_graph().get_tensor_by_name('embeddings:0')
            train_flag = tf.get_default_graph().get_tensor_by_name('phase_train:0')

            for idx, images in enumerate(images_couple):
                images = load_data(images, Config.image_size)
                feed_dict = {images_placeholder:images,train_flag:False}
                emb = sess.run(embeddings,feed_dict=feed_dict)
                dist = np.sum(np.square(np.subtract(emb[0], emb[1])), axis=0)
                print(dist)
                if dist < 1.54:
                    labels.append(1)
                else:
                    labels.append(0)

    with open('label.txt', 'w+') as f:

        for label in labels:
            f.write(str(label) + '\n')





if __name__ == '__main__':
    # draw_ROC()
    # test()
    labels = []
    with open('label.txt') as f:
        for line in f:
            # print(type(line))
            labels.append(int(line.strip()))

    print(labels.count(1))