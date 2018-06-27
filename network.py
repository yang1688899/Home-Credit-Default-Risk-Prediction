import tensorflow as tf

def weights_variable(shape, stddev=0.1, name=None):
    initial = tf.truncated_normal(shape=shape,stddev=stddev)
    return tf.Variable(initial,name=name)

def bias_variable(shape,bias=0.1, name=None):
     initial = tf.constant(bias,shape=shape)
     return tf.Variable(initial, name=None)

def network(x,keep_prob):
    fc1_w = weights_variable([65,1000])
    fc1_b = bias_variable([1000])
    fc1 = tf.nn.relu( tf.matmul(x,fc1_w)+fc1_b )

    fc1_drop = tf.nn.dropout(fc1,keep_prob=keep_prob)

    fc2_w = weights_variable([1000,1000])
    fc2_b = bias_variable([1000])
    fc2 = tf.nn.relu( tf.matmul(fc1_drop,fc2_w)+fc2_b )

    fc2_drop = tf.nn.dropout(fc2, keep_prob=keep_prob)

    fc3_w = weights_variable([1000,50])
    fc3_b = bias_variable([50])
    fc3 = tf.nn.relu( tf.matmul(fc2_drop,fc3_w)+fc3_b )

    fc4_w = weights_variable([50,1])
    fc4_b = bias_variable([1])
    logit = tf.matmul(fc3,fc4_w)+fc4_b

    return logit