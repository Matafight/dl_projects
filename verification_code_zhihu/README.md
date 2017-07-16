## Requirements:
1. python2.7 for generate_img.py
2. python3.x for cnn.py
3. tensorflow 1.2.0


## 代码解读
### logging类的工作方式

-  输出到控制台
```python
import logging
logger = logging.getLogger("starting...")
logger.setLevel(loggin.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

...
#something you want to output
logger.info("xxxx")

```

###  tensorflow中输入数据的工作方式

[Reading data:](https://www.tensorflow.org/versions/r0.12/how_tos/reading_data/)
一共有三种方式将数据输入到tensorflow模型中：

-  Feeding
-  Reading from files
-  Preloaded data

这里重点介绍Reading from files

一个典型的input pipeline 由以下几个部分组成:
1. The list of filenames
将列表形式的image_names和labels转换成tensor格式
```python
images_tensor = tf.convert_to_tensor(self.image_names, dtype=tf.string)
labels_tensor = tf.convert_to_tensor(self.labels, dtype=tf.int64)
```
2. filename shuffling(optional)
3. epoch limit(optional)
4. Filename queue 

step2,3,4可以整合到一个函数中，这个函数生成一个input队列，具体地对于slice_input_producer这个函数，假设输入的tensor中有n个样本，这个函数返回一个样本，这个队列一个样本一个样本地输出。
```python
input_queue = tf.train.slice_input_producer([images_tensor, labels_tensor], num_epochs=num_epochs)
```
5. A Reader for the file format
```python
images_content = tf.read_file(input_queue[0])
```
6. A decoder for a record read by the reader
```python
images = tf.image.convert_image_dtype(tf.image.decode_jpeg(images_content, channels=1), tf.float32)
```
7. preporcessing(optional)
比如归一化,调整大小等预处理操作
```python
new_size = tf.constant([FLAGS.image_height, FLAGS.image_width], dtype=tf.int32)
images = tf.image.resize_images(images, new_size)
```
8. Batching
```python
image_batch, label_batch = tf.train.shuffle_batch([images, labels], batch_size=batch_size, capacity=50000,
                                                          min_after_dequeue=10000)
```
9. Creating threads to prefetch using QueueRuner objects
推荐的写法如下：
```python
# Create the graph, etc.
init_op = tf.global_variables_initializer()

# Create a session for running operations in the Graph.
sess = tf.Session()

# Initialize the variables (like the epoch counter).
sess.run(init_op)

# Start input enqueue threads.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

try:
    while not coord.should_stop():
        # Run training steps or whatever
        sess.run(train_op)

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    # When done, ask the threads to stop.
    coord.request_stop()

# Wait for threads to finish.
coord.join(threads)
sess.close()
```
在sess.run()之前一定要执行tf.train.start_queue_runners()来将队列中的filename推送到reader中，不然reader会一直等待队列的输入。







### 该模型对应的tensorboard中的graph

## 新人注意事项
1. 传入num_epochs参数时，需要执行 sess.run(tf.local_variables_initializer())初始化
2. 使用try...except语句时，最好指定except希望捕获的错误类型，不然可能会导致所有的错误都被except捕获，从而导致命令行上的提醒不全面。
