import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import numpy as np

#训练数据
x = tf.placeholder(dtype=tf.float32,shape=[None,1],name="training_data")
y = tf.placeholder(dtype=tf.float32,shape=[None,1],name="training_label")

#model

Layer1 = tf.layers.Dense(units=10,activation=tf.nn.relu,use_bias=True)(inputs=x)
#Layer2 = tf.layers.Dense(units=20,activation=tf.nn.sigmoid,use_bias=True)(inputs=Layer1)
# Layer3 = tf.layers.Dense(units=300,activation=tf.nn.sigmoid,use_bias=True)(inputs=Layer2)
prediction_y = tf.layers.Dense(units=1)(inputs=Layer1)

#训练方法
loss = tf.losses.mean_squared_error(labels = y, predictions = prediction_y)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.0000001)
target = optimizer.minimize(loss)


with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    dataset = {}
    with open("Data/Gausian/linear_regression.pickle","rb") as f:
        dataset = pickle.load(f)

    #训练超参数
    epoch = 1000
    batch_size = 4
    batch_num = int(len(dataset["training_data"])/batch_size)

    for i in range(epoch):
        for j in range(batch_num):
            low = j*batch_size
            high = (j+1)*batch_size
        #开始训练
            _,losss = sess.run((target,loss),feed_dict={x:dataset["training_data"][low:high],y:dataset["training_label"][low:high]})

        print("epoch:",i,"mean_square_loss:",losss)


    #Draw  
    for i in range(100):
        plt.plot(dataset["training_data"][i][0],dataset["training_label"][i][0],"x",color="blue")

    
    predictions= sess.run(prediction_y,feed_dict={x:dataset["training_data"]})
    
    for i in range(100):
        plt.plot(dataset["training_data"][i][0],predictions[i],"x",color="red")
    # predictions = []
    # for i in range(500):
    #     p = sess.run(prediction_y,feed_dict={x:np.reshape(i,(1,1))})
    #     predictions.append(p[0][0])
    
    # x = np.linspace(0,500,500)
    # plt.plot(x,np.reshape(predictions,(500,1)),"x",color = "red")
    plt.show()