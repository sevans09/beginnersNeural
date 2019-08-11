import tensorflow as tf
import numpy as np
from tensorflow import keras

# single neuron network
model = keras.Sequential([keras.layers.Dense(units = 1, input_shape = [1])])

# stochastic gradient descent optimizer
model.compile(optimizer = 'sgd', loss = 'mean_squared_error')

# numpy array
xData = np.array([-3, 0, 3, 6, 9, 12], dtype = int)
yData = np.array([-7, 0, 5, 11, 17, 23], dtype = int)

# training
model.fit(xData, yData, epochs = 400)

# testing
print(model.predict([200]))
print(model.predict([-16]))
print(model.predict([44]))
