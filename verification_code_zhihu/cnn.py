#_*_ coding:utf-8 _*_
import tensorflow as tf
import os
import random
import logging
import time

logger = logging.getLogger('starting....')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

tf.app.flags.DEFINE_boolean('random_brightness',True,'whether to adjust brightness')
tf.app.flags.DEFINE_integer('image_width',56,'the width of images')
tf.app.flags.DEFINE_integer('image_height',28,'the height of images')
tf.app.flags.DEFINE_integer('max_steps',10000,'the max training steps')
#eval_step?
tf.app.flags.DEFINE_integer('eval_steps',100,'the step num to eval')
tf.app.flags.DEFINE_integer('save_steps',100,'the steps to save')
tf.app.flags.DEFINE_string('checkpoint_dir','./checkpoint/','the checkpoint dir')
tf.app.flags.DEFINE_string('train_data_dir','../inputs/verification_code_imgs/train_data/','training data dir')
tf.app.flags.DEFINE_string('test_data_dir','../inputs/verification_code_imgs/test_data/','testing data dir')
tf.app.flags.DEFINE_boolean('restore',False,'whether to restore from checkpoint')
tf.app.flags.DEFINE_integer('batch_size',100,'the batch_size during training')
tf.app.flags.DEFINE_string('mode','train','the run mode')
tf.app.flags.DEFINE_integer('epoch',10,'number of epoch')
FLAGS = tf.app.flags.FLAGS

#数据迭代器
class DataIterator:
    def __init__(self, data_dir):
        self.image_names = []
        for root, sub_folder, file_list in os.walk(data_dir):
            self.image_names += [os.path.join(root, file_path) for file_path in file_list]
        random.shuffle(self.image_names)
        self.labels = [self.get_label(file_name.split('/')[-1].split('.')[0].split('_')[-1]) for file_name in self.image_names]
        print(len(self.labels))
    @property
    def size(self):
        return len(self.labels)

    def get_label(self, str_label):
        """
        Convert the str_label to 10 binary code, 385 to 0001010010
        """
        result = [0]*2
        for i in str_label:
            result[int(i)] = 1
        return result

    @staticmethod
    def data_augmentation(images):
        if FLAGS.random_brightness:
            images = tf.image.random_brightness(images, max_delta=0.3)
        return images

    def input_pipeline(self, batch_size, num_epochs=None, aug=False):
        images_tensor = tf.convert_to_tensor(self.image_names, dtype=tf.string)
        labels_tensor = tf.convert_to_tensor(self.labels, dtype=tf.int64)
        input_queue = tf.train.slice_input_producer([images_tensor, labels_tensor], num_epochs=num_epochs)

        labels = input_queue[1]
        images_content = tf.read_file(input_queue[0])
        images = tf.image.convert_image_dtype(tf.image.decode_jpeg(images_content, channels=1), tf.float32)
        if aug:
            images = self.data_augmentation(images)
        new_size = tf.constant([FLAGS.image_height, FLAGS.image_width], dtype=tf.int32)
        images = tf.image.resize_images(images, new_size)
        image_batch, label_batch = tf.train.shuffle_batch([images, labels], batch_size=batch_size, capacity=50000,
                                                          min_after_dequeue=10000)
        return image_batch, label_batch

def network():
    images = tf.placeholder(dtype = tf.float32,shape=[None,28,56,1],name="image_batch")
    labels=tf.placeholder(dtype = tf.int32,shape=[None,2],name='label_batch')
    endpoints = {}
    kernel_1 = tf.get_variable('kernel_1',[5,5,1,32],tf.float32)
    conv_1 = tf.nn.conv2d(images,kernel_1,[1,1,1,1],padding = 'SAME')
    avg_pool_1 = tf.nn.avg_pool(conv_1,[1,2,2,1],[1,1,1,1],padding='SAME')
    kernel_2 = tf.get_variable('kernel_2',[5,5,32,32],tf.float32)
    conv_2 = tf.nn.conv2d(avg_pool_1,kernel_2,[1,1,1,1],padding = 'SAME')
    #avg_pool do not support max_pooling of depth
    avg_pool_2 = tf.nn.avg_pool(conv_2,[1,2,2,1],[1,1,1,1],padding = 'SAME')
    flatten = tf.contrib.layers.flatten(avg_pool_2)
    #activation_fn = tf.nn.relu
    fc1 = tf.contrib.layers.fully_connected(flatten,512,activation_fn=None)
    out0 = tf.contrib.layers.fully_connected(fc1,2,activation_fn = None)
    out1 = tf.contrib.layers.fully_connected(fc1,2,activation_fn = None)
    #global_step is used as a counter to count the times that adamoptimizer has beed called,so, it doesn't mean the num_epochs
    global_step = tf.Variable(initial_value = 0)
    out0_argmax = tf.expand_dims(tf.argmax(out0,1),1)
    out1_argmax = tf.expand_dims(tf.argmax(out1,1),1)
    out_score = tf.concat([out0,out1],axis=1)
    out_final = tf.cast(tf.concat([out0_argmax,out1_argmax],axis=1),tf.int32)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out0,labels = tf.one_hot(labels[:,0],depth = 2)))
    loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out1,labels = tf.one_hot(labels[:,1],depth = 2)))
    losses = [loss,loss1]
    loss_sum = tf.reduce_sum(losses)
    train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_sum,global_step = global_step)
    accuracy = tf.reduce_mean(tf.cast(tf.reduce_all(tf.equal(out_final,labels),axis=1),tf.float32))
    #summary.scalar 
    tf.summary.scalar('loss_sum',loss_sum)
    tf.summary.scalar('accuracy',accuracy)
    merged_summary_op = tf.summary.merge_all()

    endpoints['global_step'] = global_step
    endpoints['images'] = images
    endpoints['labels'] = labels
    endpoints['train_op'] = train_op
    endpoints['loss_sum'] = loss_sum
    endpoints['accuracy'] = accuracy
    endpoints['merged_summary_op'] = merged_summary_op
    endpoints['out_final'] = out_final
    endpoints['out_score'] = out_score
    #the following code is used for test
    print(out0_argmax.get_shape())
    print(tf.argmax(out0,1).get_shape())
    print(out0.get_shape())
    return endpoints

def train():
    train_feeder = DataIterator(data_dir = FLAGS.train_data_dir)
    test_feeder = DataIterator(data_dir = FLAGS.test_data_dir)
    
    with tf.Session() as sess:
        train_images,train_labels = train_feeder.input_pipeline(batch_size = FLAGS.batch_size,num_epochs = FLAGS.epoch)
        test_images,test_labels = test_feeder.input_pipeline(batch_size = FLAGS.batch_size)
        #num_epoches 这个参数需要通过tf.local_vairables_initializer()这个函数来初始化
        endpoints = network()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        saver = tf.train.Saver()
        start_step = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        #用于tensorboard可视化
        train_writer = tf.summary.FileWriter('./log',sess.graph)
        logger.info("training start.....")

        #checkpoint is used to save and restore models from local dick 
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                saver.restore(sess,ckpt)
                print("restoring from the checkpint {0}".format(ckpt))
        try:
            while not coord.should_stop():
                start_time = time.time()
                train_images_batch, train_labels_batch = sess.run([train_images,train_labels])
                feed_dict = {endpoints['images']:train_images_batch,endpoints['labels']:train_labels_batch}
                _,loss_val,train_summary,step = sess.run([endpoints['train_op'],endpoints['loss_sum'],endpoints['merged_summary_op'],endpoints['global_step']],feed_dict=feed_dict)
                #evaluate performance on the test data
                #batch size of test data should be just the size of test data
                #test_images
                #feed_dict_test = {endpoints['image']:test}
                logger.info("[train] the step {0} takes {1} loss {2}".format(step,time.time()-start_time,loss_val))
                end_time = time.time()
                if step%FLAGS.eval_steps == 0:
                    logger.info("========begin eval stage========")
                    start_time = time.time()
                    test_images_batch,test_labels_batch = sess.run([test_images,test_labels])
                    feed_dict_test = {endpoints['images']:test_image_batch,endpoints['labels']:test_labels_batch}
                    #test_summary 可能是用来保存到tensorborad中分析的数据？
                    accuracy,test_summary = sess.run([endpoints['accuracy'],endpoints['merged_summary_op']],feed_dict= feed_dict_test)
                    end_time = time.time()
                    logger.info('[test] the step {0}, accuracy {1}, spend time {2}'.format(step,accuracy,end_time-start_time))



                if step%FLAGS.save_steps == 0:
                    logger.info("save model in step {0}".format(step))
                    saver.save(sess,os.path.join(FLAGS.checkpoint_dir,'my_model'),global_step = endpoints['global_step'])
        except tf.errors.OutOfRangeError:
            logger.info('========training finished========')
        finally:
            coord.request_stop()
        coord.join(threads)




if __name__ == '__main__':
    train()
    #network()
