import tensorflow as tf

import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


#训练数据
x = tf.placeholder(dtype=tf.float32,shape=[None,1],name="training_data")
y = tf.placeholder(dtype=tf.float32,shape=[None,1],name="training_label")

#训练参数
weight = tf.get_variable(shape=[1,1],name="weight")
bias = tf.get_variable(shape=[1],name="bias")
prediction_y = )tf.matmul(x,weight + bias

#训练方法
loss = tf.losses.mean_squared_error(labels = y,predictions = prediction_y)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.0000001)
target = optimizer.minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    dataset = {}
    with open("Data/Gausian/linear_regression.pickle","rb") as f:
        dataset = pickle.load(f)

#训练超参数
    epoch = 100
    batch_size = 10
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
    for i in range(100):
        plt.plot(dataset["training_data"][i][0],dataset["training_label"][i][0],"x",color="blue")


    # for j in range(batch_num):
    #         low = j*batch_size
    #         high = (j+1)*batch_size
    #     #开始训练
    #         _,losss,weights,biass = sess.run((target,loss,weight,bias),
    #                     feed_dict={x:dataset["training_data"][low:high],y:dataset["training_label"][low:high]})

    #     print("epoch:",i,"mean_square_loss:",losss,weights,biass)

    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(-10, 10, 0.1)
    Y = np.arange(-10, 10, 0.1)
    X, Y = np.meshgrid(X, Y)

    losss = []
    for i in range(len(X)):
        for j in range(len(Y)):
            w = np.reshape(X[i][i],(1,1))
            b = np.reshape(Y[j][j],(1))
            losss.append(sess.run(loss,feed_dict={x:dataset["training_data"][low:high],y:dataset["training_label"][low:high],weight:w,bias:b}))
    losss = np.reshape(losss,(200,200))
    ax.plot_surface(X, Y, losss, rstride=1, cstride=1, cmap='rainbow')

    # x = np.linspace(-10, 500, 100)
    # y = weights[0]*x+biass[0]
    # plt.plot(x,y)

    plt.show()
