import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("Data/MNIST/", one_hot=True)

#训练数据
x = tf.placeholder(dtype=tf.float32,shape=[None,784],name="training_data")
y = tf.placeholder(dtype=tf.float32,shape=[None,10],name="training_label")

#model
Layer1 = tf.layers.Dense(units=30,activation=tf.nn.sigmoid,use_bias=True)(inputs=x)
Layer2 = tf.layers.Dense(units=10,activation=tf.nn.sigmoid,use_bias=True)(inputs=Layer1)


prediction_y = tf.nn.softmax(logits=Layer2,name="prediction")
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=prediction_y,labels=y)

#训练方法
correct_prediction = tf.equal(tf.argmax(prediction_y,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
target = optimizer.minimize(cross_entropy)


with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(50)
        sess.run(target, feed_dict={x: batch_xs, y: batch_ys})
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("epochs:",i,"acc:",acc)
