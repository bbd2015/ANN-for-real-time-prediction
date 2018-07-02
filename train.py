# Visualize training history
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.layers import Dropout
from keras.optimizers import RMSprop
from keras import optimizers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy
import keras

# Remove warnings
import warnings
warnings.filterwarnings('ignore')

# decide dataset
import pandas as pd

#users

# n-back
user3_n = pd.read_csv("OUT/p003-p1-n/CognitiveLoad")
user4_n = pd.read_csv("OUT/p004-p1-n/CognitiveLoad")
user6_n = pd.read_csv("OUT/p006-p1-n/CognitiveLoad")
user8_n = pd.read_csv("OUT/p008-p1-n/CognitiveLoad")

three_n = pd.read_csv("OUT/master/three_n")

# master n-back file
master_n = pd.read_csv("OUT/master/CognitiveLoadMasterN")

# card sorting
user3_c = pd.read_csv("OUT/p003-p1-s/CognitiveLoad")
user4_c = pd.read_csv("OUT/p004-p1-s/CognitiveLoad")
user6_c = pd.read_csv("OUT/p006-p1-s/CognitiveLoad")
user8_c = pd.read_csv("OUT/p008-p1-s/CognitiveLoad")

# master sorting file
master_s = pd.read_csv("OUT/master/CognitiveLoadMasterS")

# test files
test1 = pd.read_csv("arithmetic.csv")


# decide which dataset to load
dataset = master_n

# adding column labels
dataset.columns=['Timestamp','x1','x2','x3','Oxy1','DeOxy1','Oxy2','DeOxy2','Oxy3','DeOxy3', 'Oxy4','DeOxy4','Oxy5','DeOxy5','Oxy6','DeOxy6','Oxy7','DeOxy7','Oxy8','DeOxy8', 'CognitiveLoad']



# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# Remove missing values
dataset = dataset.dropna()

# shuffle the data
dataset = dataset.sample(frac=1)

# Oxy channel one and two
#Oxy1_2 = np.array(dataset["Oxy2"], dataset["Oxy3"])

OxyDeOxy1_8 = np.column_stack((dataset["Oxy1"], dataset["Oxy2"], dataset["Oxy3"], dataset["Oxy4"], dataset["Oxy5"], dataset["Oxy6"], dataset["Oxy7"], dataset["Oxy8"], dataset["DeOxy1"], dataset["DeOxy2"], dataset["DeOxy3"], dataset["DeOxy4"], dataset["DeOxy5"], dataset["DeOxy6"], dataset["DeOxy7"], dataset["DeOxy8"] ))

#Oxy1_2 = np.column_stack((dataset["Oxy2"], dataset["Oxy2"]))

# using Oxygenation channel
# input shape is the number of variables inside the X array
X = OxyDeOxy1_8
#X = np.array(dataset.ix[:,4], dataset.ix[:,5], dataset.ix[:,6], dataset.ix[:,7], dataset.ix[:,8], dataset.ix[:,9], dataset.ix[:,10], dataset.ix[:,11], dataset.ix[:,12], dataset.ix[:,13], dataset.ix[:,14], dataset.ix[:,15], dataset.ix[:,16], dataset.ix[:,17], dataset.ix[:,18], dataset.ix[:,19])
Y = np.array([[1,0] if i == 0 else [0,1] for i in dataset.CognitiveLoad])


# Dropout - the number of neurons removed at each layers, who are readded when testing
# Batch size - the number of data points added at each time, affects training time
# Epochs - the number of training/test sessions

# create model
model = Sequential()

# batchnormalization, makes the value fit between 0-1
model.add(BatchNormalization(input_shape=(16, )))
model.add(Dropout(0.5))
model.add(Dense(100, init="normal", activation='relu'))
model.add(Dense(100, init="normal", activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, init="normal", activation='relu'))
model.add(Dense(100, init="normal", activation='relu'))
#model.add(Dense(100, init='uniform', activation='relu'))
#model.add(Dense(100, kernel_initializer='uniform', activation='relu'))
#model.add(Dense(100, kernel_initializer='uniform', activation='relu'))
#model.add(Dense(100, kernel_initializer='uniform', activation='relu'))

# output layer guesses low or high cognitive load
model.add(Dense(2, init="normal", activation='softmax'))
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# Fit the model
history = model.fit(X, Y, validation_split=0.3, nb_epoch=300, batch_size=50, verbose=1)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


from keras.models import model_from_json
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
