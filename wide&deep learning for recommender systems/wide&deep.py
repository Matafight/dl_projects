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

    def __init__(self,train_path=None,test_path = None):
        self.train_path = train_path
        self.test_path = test_path
        self.k=5

    def read_data(self):
        train_data = np.loadtxt(self.train_path)
        test_data = np.loadtxt(self.test_path)
        #分别得到电影和用户的最大值，作为input dimension
        pass

    
    def dtype(self):
        return tf.float32
    
    #有多个独立的embedding layer
    def embedding_layer(self):
        #fir dim refers to batch_size
        movie_input = tf.placeholder(dtype = self.dtype(),shape=[None,self.movie_dim],name='movie_embedding')
        user_input = tf.placeholder(dtype=self.dtype(),shape=[None,self.user_dim],name = 'user_embedding')
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
        return total_bias,ele_product,
    
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
        total_bias,ele_product = self.fm_network()
        deep_layer = self.deep_network()
        #get_shape(total_bias)
        #get_shape(ele_product)
        #get_shape(deep_layer)
        comb = tf.concat([total_bias,ele_product,deep_layer],axis=1)

    def run(self):
        self.combine_network()



if __name__=='__main__':
    fm = deep_fm()
    fm.run()



    
