from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
import pandas as pd
import numpy as np
import numpy
import keras

#

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

# decide which dataset to load
dataset = user4_n

# adding column labels
dataset.columns=['Timestamp','x1','x2','x3','Oxy1','DeOxy1','Oxy2','DeOxy2','Oxy3','DeOxy3', 'Oxy4','DeOxy4','Oxy5','DeOxy5','Oxy6','DeOxy6','Oxy7','DeOxy7','Oxy8','DeOxy8', 'CognitiveLoad']


# later...
 
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


# adding x and y
dataset = user3_n

# adding column labels
dataset.columns=['Timestamp','x1','x2','x3','Oxy1','DeOxy1','Oxy2','DeOxy2','Oxy3','DeOxy3', 'Oxy4','DeOxy4','Oxy5','DeOxy5','Oxy6','DeOxy6','Oxy7','DeOxy7','Oxy8','DeOxy8', 'CognitiveLoad']

Oxy1_2 = np.column_stack((dataset["Oxy1"], dataset["Oxy2"], dataset["Oxy3"], dataset["Oxy4"], dataset["Oxy5"], dataset["Oxy6"], dataset["Oxy7"], dataset["Oxy8"], dataset["DeOxy1"], dataset["DeOxy2"], dataset["DeOxy3"], dataset["DeOxy4"], dataset["DeOxy5"], dataset["DeOxy6"], dataset["DeOxy7"], dataset["DeOxy8"] ))

X = Oxy1_2
Y = np.array([[1,0] if i == 0 else [0,1] for i in dataset.CognitiveLoad])


# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

# calculate predictions
predictions = loaded_model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)




