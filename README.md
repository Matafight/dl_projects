# dl_projects
Play with deep learning

# Anaconda 配置tensorflow 以及python2和python3共存
## 安装tensorflow
安装非GPU版本的tensorflow很简单
```bash
conda create -n tensorflow python=3.5
activate tensorflow
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-1.2.1-cp35-cp35m-win_amd64.whl 
```
## python版本共存
有时候需要在网上找一些小脚本用来执行一些命令，可能会出现python2和python3不兼容的问题，尤其是字符串编码问题，甚是头疼，为了方便起见，考虑既装python2又装python3。可以不用装两个anaconda，装一个就行。
假设首先装一个Anaconda3，之后在python3的anaconda中隔离一个环境装python2
```python
conda create -n python2 python=2.7 [anaconda]
```
后面的anaconda选项可以加或不加，不过加的话就要重新安装非常多的anaconda中自带的package，我感觉还不如重新下载一个anaconda2安装呢。不加的话就是安装了基本的python2.7，没有那么多科学计算包，不过可以针对性地使用pip安装有需要的package。
想使用python2.7的时候，只需要在命令行中执行
```bash
activate python2
```

# 使用TensorBoard做可视化

需要在程序中添加如下代码,将图的信息存到文件中：
```python
#sess = tf.Session()
train_writer = tf.summary.FileWriter('./logs',sess.graph)
```
tensorboard --logdir=logs

这样写是错的:
tensorboard --logdir='./logs'
tensorboard --logdir = logs (连空格都不能加？目前我测试是这样的，至少当我加了空格之后tensorboard中的graphs就为空了，显示No graph definition files were found)
