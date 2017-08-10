#_*_ coding:utf-8_*_
#implement wide&deep learning for recommender systems using tensorflow without using the blackbox function tf.contrib.learn.DNNLinearCombinedClassifier
# steps for constructing graph
# step1. inference
# step2. loss
# step3. training


import tensorflow as tf
import numpy as np

def get_shape(tensor):
    print(tensor.get_shape())

class deep_fm:
# read data
    movie_dim = 100
    user_dim = 100
    k = 0
    hidden1 = 3
    hidden2 = 3
    train_X = None
    test_X = None
    train_y = None
    test_y = None 
    train_user_onehot = None
    train_movie_onehot = None
    test_user_onehot = None
    test_movie_onehot = None
    batch_size = None



    def __init__(self,train_path=None,test_path = None):
        self.train_path = train_path
        self.test_path = test_path
        self.k=5

    def read_data(self):
        train_data = np.loadtxt(self.train_path)
        test_data = np.loadtxt(self.test_path)
        self.batch_size = train_data.shape[0]
        print(train_data.shape)
        total_data = np.concatenate([train_data,test_data],axis=0)
        self.movie_dim = np.amax(total_data[:,0])
        #self.user_dim = np.amax(total_data[:,1])
        self.user_dim = self.movie_dim

        self.train_X = train_data[:,0:2]
        self.test_X = test_data[:,0:2]
        self.train_y = train_data[:,2].reshape(train_data.shape[0],1)
        self.test_y = test_data[:,2].reshape(test_data.shape[0],1)
        
        self.train_user_onehot = tf.one_hot(indices = self.train_X[:,0],depth = self.user_dim)
        self.train_movie_onehot = tf.one_hot(indices=self.train_X[:,1],depth = self.movie_dim)
        self.test_user_onehot = tf.one_hot(indices = self.test_X[:,0],depth = self.user_dim)
        self.test_movie_onehot = tf.one_hot(indices=self.test_X[:,1],depth = self.movie_dim)

        #one hot 编码，分别存储user 和 movie
        #分别得到电影和用户的最大值，作为input dimension

    
    def dtype(self):
        return tf.float32
    
    #有多个独立的embedding layer
    def embedding_layer(self):
        #fir dim refers to batch_size
        movie_input = tf.placeholder(dtype = self.dtype(),shape=[self.batch_size,self.movie_dim],name='movie_embedding')
        user_input = tf.placeholder(dtype=self.dtype(),shape=[self.batch_size,self.user_dim],name = 'user_embedding')
        #var_movie = tf.get_variable(name='var_movie_embedding',shape=[self.movie_dim,self.k],dtype=self.dtype())
        #var_user = tf.get_variable(name='var_user_embedding',shape=[self.user_dim,self.k],dtype = self.dtype())

        #fc_movie
        fc_movie = tf.contrib.layers.fully_connected(movie_input,self.k,activation_fn = None)
        fc_user = tf.contrib.layers.fully_connected(user_input,self.k,activation_fn = None)
        return movie_input,user_input,fc_movie,fc_user

    def fm_network(self):
        movie_input,user_input,fc_movie,fc_user =  self.embedding_layer()
        #bias term 
        #movie_bias = tf.get_variable(name='movie_bias',shape=[self.movie_dim,1],dtype = self.dtype())
        #user_bias = tf.get_variable(name='user_bias',shape=[self.user_dim,1],dtype = self.dtype())
        #total_bias = tf.matmul(movie_input,movie_bias)+ tf.matmul(user_input,user_bias)
        movie_bias = tf.contrib.layers.fully_connected(movie_input,1,activation_fn= None)
        user_bias = tf.contrib.layers.fully_connected(user_input,1,activation_fn = None)
        total_bias = movie_bias + user_bias
        #inner produce between fc_movie and fc_user
        ele_product = tf.multiply(fc_movie,fc_user)
        # I need to inspect the shape of ele_product
        #out_fm = tf.reduce_sum(ele_product,axis=1) + tf.reduce_sum(movie_input)+tf.reduce_sum(user_input)
        #return out_fm
        return total_bias,ele_product,movie_input,user_input
    
    def deep_network(self):
        movie_input,user_input,fc_movie,fc_user = self.embedding_layer()
        #var_movie_sec = tf.get_variable(name='var_movie_sec',shape=[self.k,self.hidden1],dtype = self.dtype())
        #var_user_sec =tf.get_variable(name='var_user_sec',shape=[self.k,self.hidden1],dtype = self.dtype())
        #sum_movie = tf.matmul(fc_movie,var_movie_sec)
        #sum_user = tf.matmul(fc_user,var_user_sec)
        sum_movie = tf.contrib.layers.fully_connected(fc_movie,self.hidden1,activation_fn = None)
        sum_user = tf.contrib.layers.fully_connected(fc_user,self.hidden1,activation_fn = None)

        sum_all = sum_movie+sum_user
        out_sec = tf.nn.relu(sum_all)

        out_third = tf.contrib.layers.fully_connected(out_sec,num_outputs = self.hidden2)
        #deep_layer = tf.contrib.layers.fully_connected(out_third,num_outputs = 1,activation = tf.nn.sigmoid)
        return out_third
    def combine_network(self):
        total_bias,ele_product,movie_input,user_input = self.fm_network()
        deep_layer = self.deep_network()
        #comb = tf.concat([total_bias,ele_product,deep_layer],axis=1)
        comb = deep_layer
        output = tf.contrib.layers.fully_connected(comb,num_outputs = 1,activation_fn = None)
        #当作一个回归问题，
        y = tf.placeholder(dtype = self.dtype(),shape=[None,1],name='labels')
        loss = tf.losses.mean_squared_error(labels = y,predictions = output)
        #获取trainable变量的梯度，并传入update中
        trainable_v = tf.trainable_variables()
        opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)   
        grads_and_vars = opt.compute_gradients(loss,trainable_v)       
        ret_opt = opt.apply_gradients(grads_and_vars)

        endpoints={}
        endpoints['movie_input'] = movie_input
        endpoints['user_input'] = user_input
        endpoints['labels'] = y
        endpoints['losses'] = loss
        endpoints['train_op'] = ret_opt
        return endpoints

    def run(self):
        #准备数据
        self.read_data()
        endpoints = self.combine_network()

        with tf.Session() as sess:
            #作为测试，现在每次都feed进去所有的数据
            #train_movie_onehot = tf.expand_dims(self.train_movie_onehot,-1)
            #train_user_onehot = tf.expand_dims(self.train_user_onehot,-1)
            sess.run(tf.global_variables_initializer())     
            sess.run(tf.local_variables_initializer())
            train_movie_onehot = self.train_movie_onehot
            train_user_onehot = self.train_user_onehot
            train_movie_onehot = tf.cast(train_movie_onehot,tf.float32)
            movie_input = sess.run(train_movie_onehot)
            print('movie shape:')
            print(movie_input.shape)
            user_input = sess.run(train_user_onehot)
            print(user_input.shape)
            feed_dict_loc = {endpoints['movie_input']:movie_input,endpoints['user_input']:movie_input,endpoints['labels']:self.train_y}
            #_,losses = sess.run([endpoints['train_op'],endpoints['losses']],feed_dict=feed_dict_loc)
            #losses = sess.run(endpoints['losses'],feed_dict=feed_dict_loc)
            endpoints['losses'].eval(feed_dict = feed_dict_loc)






if __name__=='__main__':
    train_path = 'u1.base.txt'
    test_path = 'u1.test.txt'
    fm = deep_fm(train_path,test_path)
    fm.run()



    
