import numpy as np
import matplotlib.pyplot as plt
import pickle as pk

x_cross = []
y_cross = []
x_circle = []
y_circle = []

for i in range(500):
    x_cross.append(np.random.normal(3,30))
    y_cross.append(np.random.normal(3,30))
    x_circle.append(np.random.normal(75,20))
    y_circle.append(np.random.normal(75,20))

dataset = {"training_data":[],"validation_data":[],"training_label":[],"validation_label":[]}

for i in range(500):
    dataset["training_data"].append([x_circle[i],y_circle[i]])
    dataset["training_label"].append([1])
    
    dataset["training_data"].append([x_cross[i],y_cross[i]])
    dataset["training_label"].append([0])


with open("Data/Gausian/linear_classification.pickle","wb") as f:
    pk.dump(dataset,f)

with open("Data/Gausian/linear_regression.pickle","wb") as f:
    x = np.linspace(1, 500, 100)
    y = 0.7*x + 0.3 + np.random.normal(0,10,100)
    pk.dump({"training_data":np.reshape(x,(100,1)),"training_label":np.reshape(y,(100,1))},f)

    
# for i in range(500):
#     if dataset["training_label"][i] == 【1】:
#         plt.plot(dataset["training_data"][i][0],dataset["training_data"][i][1],"o",color="blue")
#     else:
#         plt.plot(dataset["training_data"][i][0],dataset["training_data"][i][1],"x",color="red")


# x = np.linspace(0, 4, 1000)
# y = (0.0125*x-1.257)/0.0139
# plt.plot(x,y)

# plt.show()