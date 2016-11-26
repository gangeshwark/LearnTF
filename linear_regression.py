import numpy as np
import tensorflow as tf
#linear regression

#Generate random data - 2M data points
x_data = np.random.rand(2000000).astype(np.float32)
y_data = x_data * 9 + 7.8

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b
print y
loss = tf.reduce_mean(tf.square( y - y_data))
optimizer = tf.train.AdamOptimizer(0.1)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in xrange(2001):
    sess.run(train)
    if step % 10 == 0:
        print "Step: %d\t W = %f\t b = %f\t loss = %f" %(step, sess.run(W), sess.run(b), sess.run(loss))