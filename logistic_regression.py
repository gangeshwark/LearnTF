import numpy as np
import tensorflow as tf
#linear regression

#sigmoid function
def sig(x):
    return 1/(1 + np.exp(-x))


#Generate random data - 2M data points
data_points = 2000000
x_data = np.random.rand(data_points).astype(np.float32)
#y_data = sig(x_data * 9 + 7.8)
#tensorflow's inbuilt sigmoid funtion
y_data = tf.nn.sigmoid(x_data * 9 + 7.8)


W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b
print y
loss = tf.reduce_mean(tf.square( y - y_data))
learning_rate = 0.1
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in xrange(2001):
    sess.run(train)
    if step % 10 == 0:
        print "Step: %d\t W = %f\t b = %f\t loss = %f" %(step, sess.run(W), sess.run(b), sess.run(loss))