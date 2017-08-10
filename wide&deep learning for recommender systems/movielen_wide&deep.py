#coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

COLUMNS = ['users','movies']
LABEL = ['rating']
TRAINING_PATH = '../inputs/ml-100k/u1.base'
TESTING_PATH = '../inputs/ml-100k/u1.test'

def input_fn(data_path):
    data = np.loadtxt(data_path)
    row = data.shape[0]
    users = tf.constant(data[:,0].reshape(row,1),dtype = tf.float32)
    movies = tf.constant(data[:,1].reshape(row,1),dtype = tf.float32)
    rating = tf.constant(data[:,2],dtype = tf.float32)
 
    feature_cols = {'users':users,'movies':movies}
    print(users.get_shape())
    print(rating.get_shape())
    return feature_cols,rating



def model_builder():
    user_column = tf.contrib.layers.sparse_column_with_hash_bucket('users',hash_bucket_size = 1000,dtype = tf.int32)
    movie_column = tf.contrib.layers.sparse_column_with_hash_bucket('movies',hash_bucket_size = 1000,dtype = tf.int32)
    
    wide_columns = [user_column,movie_column,tf.contrib.layers.crossed_column([user_column,movie_column],hash_bucket_size = int(1e+6))]
    deep_columns = [tf.contrib.layers.embedding_column(user_column,dimension = 10),tf.contrib.layers.embedding_column(movie_column,dimension = 10)]

    m = tf.contrib.learn.DNNLinearCombinedRegressor(
        model_dir = './checkpoint/',
        linear_feature_columns = wide_columns,
        dnn_feature_columns = deep_columns,
        dnn_hidden_units = [100,50],
        fix_global_step_increment_bug = True)
    return m


def train_and_eval():
    model = model_builder()
    model.fit(input_fn = lambda:input_fn(TRAINING_PATH),steps = 1000)
    results = model.evaluate(input_fn = lambda:input_fn(TESTING_PATH),steps = 1,metrics = {'rmse':tf.metrics.root_mean_squared_error})
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))

if __name__=='__main__':
    train_and_eval()



