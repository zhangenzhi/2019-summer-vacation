import tensorflow as tf

import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


#训练数据
x = tf.placeholder(dtype=tf.float32,shape=[None,2],name="training_data")
y = tf.placeholder(dtype=tf.float32,shape=[None,1],name="training_label")

#训练参数
weight = tf.get_variable(shape=[2,1],name="weight")
bias = tf.get_variable(shape=[1],name="bias")
prediction_y = tf.sigmoid(tf.add(tf.matmul(x,weight),bias))

#训练方法
loss = tf.losses.mean_squared_error(labels = y,predictions = prediction_y)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
target = optimizer.minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    dataset = {}
    with open("Data/Gausian/linear_classification.pickle","rb") as f:
        dataset = pickle.load(f)

#训练超参数
    epoch = 1000
    batch_size = 50
    batch_num = int(len(dataset["training_data"])/batch_size)

    weights = []
    biass = []
    for i in range(epoch):
        for j in range(batch_num):
            low = j*batch_size
            high = (j+1)*batch_size
        #开始训练
            _,losss,weights,biass = sess.run((target,loss,weight,bias),
                        feed_dict={x:dataset["training_data"][low:high],y:dataset["training_label"][low:high]})

        print("epoch:",i,"mean_square_loss:",losss,weights,biass)


#Draw  
    for i in range(1000):
        if dataset["training_label"][i] == [1]:
            plt.plot(dataset["training_data"][i][0],dataset["training_data"][i][1],"o",color="blue")
        else:
            plt.plot(dataset["training_data"][i][0],dataset["training_data"][i][1],"x",color="red")


#results
    X = np.linspace(-150,150,100)
    Y = (weights[0]*X+biass[0])/(-weights[1])
    plt.plot(X,Y)



    plt.show()
