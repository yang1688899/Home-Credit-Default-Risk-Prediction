import tensorflow as tf
import network
import utils
import config
import os
from sklearn.model_selection import train_test_split
from math import ceil

x = tf.placeholder(tf.float32, [None,65])
y = tf.placeholder(tf.float32, [None,1])
keep_prob = tf.placeholder(tf.float32)

logit = network.network(x,keep_prob=keep_prob)

loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=logit,labels=y) )
train_op = tf.train.AdamOptimizer().minimize(loss)
acc = tf.reduce_mean( tf.cast(tf.equal(y,tf.round(logit)),dtype=tf.float32) )

trainfile = os.path.join(config.DATADIR,'application_train.csv')
features, labels = utils.process_data(trainfile)
features_train,features_valid,labels_train,labels_valid = train_test_split(features,labels,train_size=0.95)
features_train = utils.normlize_data_train(features_train, './scaler.p')
features_valid = utils.normlize_data_test(features_valid, './scaler.p')

batch_size = 1024
max_step = 100000
logger = utils.get_logger(os.path.join(config.LOGDIR,'info.log'))
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print("training")
    train_gen = utils.get_data_batch(features_train, labels_train, batch_size=batch_size, is_shuffle=True)
    for step in range(0, max_step):
        batch_features_train, batch_labels_train = next(train_gen)
        train_loss = sess.run(train_op,feed_dict={x:batch_features_train, y:batch_labels_train, keep_prob:0.5})

        if step%50 == 0:
            train_loss,train_accuracy = sess.run([loss,acc],feed_dict={x:batch_features_train, y:batch_labels_train, keep_prob:1.})
            logger.info('In step %s training loss is %s, training accuracy is %s'%(step,train_loss,train_accuracy))
            print('In step %s training loss is %s, training accuracy is %s'%(step,train_loss,train_accuracy))

        if step%1000 == 1:
            valid_gen = utils.get_data_batch(features_valid,labels_valid,batch_size=batch_size,is_shuffle=False)
            total_accuracy = 0
            for i in range( ceil(len(features_valid)/batch_size)):
                batch_features_valid, batch_labels_valid = next(valid_gen)
                accuracy = sess.run(acc,feed_dict={x:batch_features_valid, y:batch_labels_valid, keep_prob:1.0})
                total_accuracy += accuracy
            current_accuracy = total_accuracy/ceil(len(features_valid)/batch_size)
            saver.save(sess, config.CHECKFILE, global_step=step)

            print('The valid accuracy at step step is %s, save model a step %s'%(step,step))
            logger.info('The valid accuracy at step step is %s, save model a step %s'%(step,step))


