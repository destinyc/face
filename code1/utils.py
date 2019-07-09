import tensorflow as tf
import os
import re
import time
from tensorflow.python.platform import gfile




# 这里是求特征关于特征中心的L2损失
def center_loss(features,label,alfa,nrof_classes):
    '''

    :param features:    [batch_size, 128]  一个batch图像对应的特征
    :param label:       这个batch图像对应的label
    :param alfa:        用于更新特征中心的更新率
    :param nrof_classes:       文件中总共的人物数
    :return:             中心损失和特征中心
    '''
    # 获取特征长度 128
    nrof_features = features.get_shape()[1]
    # 建立一个变量，存储每一类的中心，不训练，但是下面会更新中心值
    centers = tf.get_variable('centers',[nrof_classes,nrof_features],dtype=tf.float32,
                              initializer=tf.constant_initializer(0),trainable=False)
    # 特征成一维
    label = tf.reshape(label,[-1])
    # 类似数组索引，把向量中某些索引值提取出来
    #获取当前batch每个样本对应的中心
    centers_batch = tf.gather(centers,label)
    diff = (1-alfa)*(centers_batch-features)
    # 将centers中label位置的数减去diff
    # 更新中心
    centers = tf.scatter_sub(centers,label,diff)
    # L2损失
    loss= tf.reduce_mean(tf.square(features-centers_batch))
    return loss,centers


#把总和加到损失上
def _add_loss_summaries(total_loss):
    #滑动平均 更新参数，衰减率0.9
    #相当于指数加权平均 β=0.9
    #β = min(β,(1+steps)/(10+steps))
    #下面三行 即为公式loss_averages_op = β* loss_averages_op_prev + (1-β)*(loss+[total_loss])
    loss_averages = tf.train.ExponentialMovingAverage(0.9,name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses+[total_loss])

    return loss_averages_op



def save_variables_and_metagraph(sess,saver,model_dir,model_name,step):
    #保存模型检查点
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir,'model-%s.ckpt'%model_name)
    # Weite_meta_graph设为False是因为图是一样的 只需要保存一次
    saver.save(sess,checkpoint_path,global_step=step,write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds'%save_time_variables)
    metagraph_filename = os.path.join(model_dir,'model-%s.meta'%model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds'%save_time_metagraph)


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    ckpt_file = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


def load_model(model):
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s'%model_exp)
        #无阻塞获取文件操作句柄,下面是导入pb文件
        with gfile.FastGFile(model_exp,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def,name='')
    else:
        print('Model directory: %s'%model_exp)
        meta_file,ckpt_file = get_model_filenames(model_exp)
        print('Metagraph file: %s'%meta_file)
        print('Checkpoint file: %s'%ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp,meta_file))
        saver.restore(tf.get_default_session(),os.path.join(model_exp,ckpt_file))