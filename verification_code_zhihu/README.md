## Requirements:
1. python2.7 for generate_img.py
2. python3.x for cnn.py
3. tensorflow 1.2.0


## 代码解读
1. logging类的工作方式

a. 输出到控制台
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

2. tensorflow中输入数据的工作方式
3. 该模型对应的tensorboard中的graph

## 新人注意事项
1. 传入num_epochs参数时，需要执行 sess.run(tf.local_variables_initializer())初始化
2. 使用try...except语句时，最好指定except希望捕获的错误类型，不然可能会导致所有的错误都被except捕获，从而导致命令行上的提醒不全面。
