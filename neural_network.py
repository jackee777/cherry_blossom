from sklearn.utils import shuffle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam

np.random.seed(0)

M = 2
K = 3
n = 100
N = n * K

model = Sequential(
    [Dense(input_dim=2, units=1),
     Activation('tanh')]
)

model.compile(loss = 'binary_crossentropy', optimizer = Adam(lr = 0.001))

model.fit(X, y, epochs = 200, batch_size = 1)
