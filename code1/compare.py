from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import tensorflow as tf
import numpy as np
import sys
import argparse
import utils
import data_process


def main(args):
    images = load_data(args.image_files, args.image_size)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            #加载模型
            utils.load_model(args.model)
            images_placeholder = tf.get_default_graph().get_tensor_by_name('batch_join:0')
            embeddings = tf.get_default_graph().get_tensor_by_name('embeddings:0')
            train_flag = tf.get_default_graph().get_tensor_by_name('phase_train:0')

            feed_dict = {images_placeholder:images,train_flag:False}
            emb = sess.run(embeddings,feed_dict=feed_dict)
            #输出emb为2 x 128 ，分别为两张图片的128维向量
            dist = np.sum(np.square(np.subtract(emb[0], emb[1])), axis = 0)
            print(dist)
            if dist <1.54:
                print("It's same person!")
            else:
                print("It's not the same one!")

def load_data(image_paths, image_size):

    img_list = []
    print(image_paths)
    for image in image_paths:
        img = cv2.imread(image)
        img = cv2.resize(img, (image_size, image_size))

        prewhitened = data_process.prewhiten(img)
        img_list.append(prewhitened)

    images = np.stack(img_list)

    return images


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',
                        default = 'model/center_softmax/71_epoch')
    parser.add_argument('image_files', type=str, nargs='+', help='Images to compare')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))