import tensorflow as tf
import network
import utils
import config
import os
from sklearn.model_selection import train_test_split

x = tf.placeholder(tf.float32, [None,65])
y = tf.placeholder(tf.float32, [None,1])
keep_prob = tf.placeholder(tf.float32)

logit = network.network(x,keep_prob=keep_prob)

loss = -tf.reduce_mean( tf.reduce_sum(y*tf.log(logit) + (1-y)*tf.log(1-logit)) )
train_op = tf.train.AdamOptimizer().minimize(loss)
acc = tf.reduce_mean( tf.cass(tf.equal(y,tf.round(logit))) )

trainfile = os.path.join(config.DATADIR,'application_train.csv')
features, labels = utils.process_data(trainfile)
features_train,features_valid,labels_train,labels_valid = train_test_split(features,labels,shuffle=True,train_size=0.95)
features_train = utils.normlize_data_train(features_train, './scaler.p')
features_valid = utils.normlize_data_test(features_valid, './scaler.p')

batch_size = 256
max_step = 100000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(0, max_step):



