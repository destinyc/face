
from datetime import datetime
import os
import time
import tensorflow as tf
import numpy as np
import importlib
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

from config import Config
import data_process
import utils


def train_batch(total_loss, global_step, optimizer, learning_rate, moving_average_decay,
          update_gradient_vars):
    loss_averages_op = utils._add_loss_summaries(total_loss)

    #计算梯度
    with tf.control_dependencies([loss_averages_op]):
        if optimizer == 'ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer == 'ADADELTA':
            opt = tf.train.AdadeltaOptimizer(learning_rate,rho=0.9,epsilon=1e-6)
        elif optimizer == 'ADAM':
            opt = tf.train.AdamOptimizer(learning_rate,beta1=0.9,beta2=0.999,epsilon=0.1)
        elif optimizer == 'RMSPROP':
            opt = tf.train.RMSPropOptimizer(learning_rate,decay=0.9,momentum=0.9,epsilon=1.0)
        elif optimizer == 'MOM':
            opt = tf.train.MomentumOptimizer(learning_rate,0.9,use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')

        grads = opt.compute_gradients(total_loss,update_gradient_vars)

    apply_gradient_op = opt.apply_gradients(grads,global_step=global_step)

    variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay,global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.control_dependencies([apply_gradient_op,variables_averages_op]):
        train_op = tf.no_op(name='train')
    return train_op

#训练一个 epoch
def train(sess,epoch,image_list,label_list,index_dequeue_op,enqueue_op,image_paths_placeholder,labels_placeholder,
          learning_rate_placeholder,train_flag,batch_size_placeholder,global_step,
          loss,train_op,regularization_losses):
    batch_number = 0
    lr = Config.learning_rate

    index_epoch = sess.run(index_dequeue_op)
    label_epoch = np.array(label_list)[index_epoch]
    image_epoch = np.array(image_list)[index_epoch]

    labels_array = np.expand_dims(np.array(label_epoch),1)
    images_paths_array = np.expand_dims(np.array(image_epoch),1)
    sess.run(enqueue_op,{image_paths_placeholder:images_paths_array,labels_placeholder:labels_array})

    train_time = 0
    while batch_number < Config.epoch_size:
        start_time = time.time()
        feed_dict = {learning_rate_placeholder:lr,train_flag:True,batch_size_placeholder:Config.batch_size}
        if batch_number % 100 ==0:
            err,_,step,reg_loss = sess.run([loss,train_op,global_step,regularization_losses],feed_dict=feed_dict)

        else:
            err,_,step,reg_loss = sess.run([loss,train_op,global_step,regularization_losses],feed_dict=feed_dict)
        duration = time.time() - start_time
        print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tRegLoss %2.3f'%
              (epoch,batch_number+1,Config.epoch_size,duration,err,np.sum(reg_loss)))
        batch_number += 1
        train_time += duration

    return step


def main():
    # 导入模型
    network = importlib.import_module(Config.model_def)                 # 相当于导入 .py 文件

    # 用时间命名
    subdir = datetime.strftime(datetime.now(),'%Y%m%d-%H%M%S')
    model_dir = os.path.join(os.path.expanduser(Config.models_base_dir),subdir)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    # 读取数据
    train_set = data_process.get_data_set(Config.data_dir)

    # 类别总数
    nrof_classes = len(train_set)

    pretrained_model = None
    if Config.pretrained_model:
        pretrained_model = os.path.expanduser(Config.pretrained_model)
        print('Pre-trained model: %s'%pretrained_model)

    with tf.Graph().as_default():
        global_step = tf.Variable(0,trainable=False)

        image_list, label_list = data_process.get_image_paths_and_labels(train_set)
        assert len(image_list)>0,'The dataset should not empty'

        labels = ops.convert_to_tensor(label_list,dtype=tf.int32)
        range_size = array_ops.shape(labels)[0]

        index_queue = tf.train.range_input_producer(range_size,num_epochs=None,shuffle=True,seed = None,capacity=32)

        index_dequeue_op = index_queue.dequeue_many(Config.batch_size*Config.epoch_size,'index_dequeue')

        learning_rate_placeholder = tf.placeholder(tf.float32,name='learning_rate')
        batch_size_placeholder = tf.placeholder(tf.int32,name='batch_size')
        train_flag = tf.placeholder(tf.bool,name='phase_train')
        image_paths_placeholder = tf.placeholder(tf.string,shape=(None,1),name='image_paths')
        labels_placeholder = tf.placeholder(tf.int64,shape=(None,1),name='labels')

        input_queue = data_flow_ops.FIFOQueue(capacity=500000,
                                              dtypes=[tf.string,tf.int64],
                                              shapes=[(1,),(1,)],
                                              shared_name=None,name=None)
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder,labels_placeholder],name='enqueue_op')

        nrof_preprocess_threads = 4
        images_and_labels = []
        for _ in range(nrof_preprocess_threads):
            filenames, label = input_queue.dequeue()
            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                image = tf.image.decode_image(file_contents, channels=3)
                if Config.random_rotate:
                    image = tf.py_func(data_process.random_rotate_image, [image], tf.uint8)
                if Config.random_crop:
                    image = tf.random_crop(image, [Config.image_size, Config.image_size, 3])
                else:
                    image = tf.image.resize_image_with_crop_or_pad(image, Config.image_size, Config.image_size)
                if Config.random_flip:
                    image = tf.image.random_flip_left_right(image)


                # pylint: disable=no-member
                image.set_shape((Config.image_size, Config.image_size, 3))
                images.append(tf.image.per_image_standardization(image))
            images_and_labels.append([images, label])

        image_batch,label_batch = tf.train.batch_join(
            images_and_labels,batch_size=batch_size_placeholder,
            shapes=[(Config.image_size,Config.image_size,3),()],enqueue_many=True,
            capacity=4*nrof_preprocess_threads*Config.batch_size,
            allow_smaller_final_batch=True)
        image_batch = tf.identity(image_batch,'image_batch')
        image_batch = tf.identity(image_batch,'input')
        label_batch = tf.identity(label_batch,'label_batch')

        print('Total number of classes: %d'%nrof_classes)
        print('Total number of examples: %d'%len(image_list))

        print('Building training graph')

        prelogits = network.inference(image_batch,Config.keep_prob,
                                        phase_train = train_flag,bottleneck_layer_size = Config.embedding_size,
                                        weight_decay = Config.weight_decay)
        logits = slim.fully_connected(prelogits, len(train_set), activation_fn=None,
                                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      weights_regularizer=slim.l2_regularizer(Config.weight_decay),
                                      scope='Logits', reuse=False)

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        # 添加中心损失
        if Config.center_loss_weight >0.0:
            prelogits_center_loss,_ = utils.center_loss(prelogits,label_batch,Config.center_loss_alfa,nrof_classes)
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,prelogits_center_loss*Config.center_loss_weight)
        learning_rate = tf.train.exponential_decay(learning_rate_placeholder,global_step,
                                                   Config.learning_rate_decay_epochs*Config.epoch_size,
                                                   Config.learning_rate_decay_factor,staircase=True)


        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_batch,logits=logits,
                                                                       name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')
        tf.add_to_collection('losses',cross_entropy_mean)

        # 把中心损失加到交叉softmax损失上
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([cross_entropy_mean]+regularization_losses,name='total_loss')

        # 一个batch 训练操作并更新模型参数
        train_op = train_batch(total_loss,global_step,Config.optimizer,learning_rate,
                               Config.moving_average_decay,tf.global_variables())

        # 创建一个保存器
        saver = tf.train.Saver(tf.trainable_variables(),max_to_keep=3)


        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = Config.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,log_device_placement=False))
        sess.run(tf.global_variables_initializer())

        # 获得线程坐标，启动填充队列的线程
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord,sess=sess)

        with sess.as_default():
            sess.run(tf.local_variables_initializer())
            if pretrained_model:
                print('Restoring pretrained model: %s'%pretrained_model)
                meta_file, ckpt_file = utils.get_model_filenames(Config.pretrained_model)
                saver = tf.train.import_meta_graph(os.path.join(Config.pretrained_model, meta_file))
                saver.restore(sess, os.path.join(Config.pretrained_model, ckpt_file))
            print('Running training')
            epoch = 0
            while epoch < Config.max_nrof_epochs:
                step = sess.run(global_step,feed_dict=None)
                utils.save_variables_and_metagraph(sess, saver, model_dir, subdir, step)

                print('++++++++++save done++++++++++')
                epoch = step // Config.epoch_size
                # 训练一个epoch
                train(sess,epoch,image_list,label_list,index_dequeue_op,enqueue_op,image_paths_placeholder,labels_placeholder,
                      learning_rate_placeholder,train_flag,batch_size_placeholder,global_step,
                      total_loss,train_op,regularization_losses)
                utils.save_variables_and_metagraph(sess,saver,model_dir,subdir,step)

    return model_dir

if __name__ =='__main__':

    main()
