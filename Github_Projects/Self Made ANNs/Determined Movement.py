import numpy as np
import Object_Oriented_Numpy_ANN as ann
from sklearn.preprocessing import OneHotEncoder


X, y = ann.create_data(100,3)
y = y.reshape(300,1)
#Convert y to one hot encoded vector

onehot = OneHotEncoder()
encoded = onehot.fit_transform(y)
y = encoded.toarray()





layer1 = ann.Layer_Dense(2,3,"ReLU")
layer2 = ann.Layer_Dense(3,3, "Softmax")


#To add: epochs
w = ann.runModel(X)


loss_function = ann.Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(w.output, y)


print('Loss:', loss)



for epoch in range(20):
    w.backpropagate(0.01, y)
    w = ann.runModel(X)
    loss = loss_function.calculate(w.output, y)
    print('Loss:', loss)

