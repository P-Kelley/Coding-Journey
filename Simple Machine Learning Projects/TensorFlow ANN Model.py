import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout


((X_train, y_train),(X_test, y_test)) = tf.keras.datasets.mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0


model = Sequential()
model.add(Flatten(input_shape =(28,28)))
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 10, activation = 'sigmoid'))

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(loss = loss_fn, optimizer = 'sgd', metrics = 'accuracy')
model.fit(X_train, y_train, epochs = 15, batch_size = 40)

print(model.evaluate(X_test, y_test))
