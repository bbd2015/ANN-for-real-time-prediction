#!/usr/bin/python
import MySQLdb
import pandas as pd
import numpy as np
from keras.models import model_from_json

# Open the file
f = open('output.csv', 'w')

db = MySQLdb.connect(host="localhost",    # your host, usually localhost
                     user="root",         # your username
                     passwd="",  # your password
                     db="newttt")        # name of the data base

# you must create a Cursor object. It will let
#  you execute all the queries you need
cur = db.cursor()

# Use all the SQL you like
cur.execute("SELECT * FROM REALTIME1")

columns = ['Oxy1','DeOxy1','Oxy2','DeOxy2','Oxy3','DeOxy3', 'Oxy4','DeOxy4','Oxy5','DeOxy5','Oxy6','DeOxy6','Oxy7','DeOxy7','Oxy8','DeOxy8']


#while True:
    # Read the data
#    df = pd.DataFrame((np.array(cur.fetchmany(1000)), columns))
    # We are done if there are no data
#    if len(df) == 0:
#        break
        # Lets write to the file
#    else:
#        df.to_csv(f, header=False)

# later...



def predict(data):
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    # predicting the model
    print("Predicting the value")
    # swap the axes back for prediction
    prediction = loaded_model.predict(np.swapaxes(np.array(data), 0, 1))
    print(loaded_model.predict(np.swapaxes(np.array(data), 0, 1)))

    if (np.floor(prediction[0][0]) == 0.0):
        print("high")
    elif (np.floor(prediction[0][0]) == 1.0):
        print("low")
    else:
        print("error")





data = [[0, 2, 3, 5, 4, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1]]
# swap the axes of the array
data = np.swapaxes(data,0,1)
print(np.array(data).shape)
predict(data)

# print all the first cell of all the rows
# for row in cur.fetchall():
#     rows = [[row[0]], [row[1]], [row[2]], [row[3]], [row[4]], [row[5]], [row[6]], [row[7]], [row[8]], [row[9]], [row[10]], [row[11]], [row[12]], [row[13]], [row[14]], [row[15]]]
#     df = pd.DataFrame(np.array(rows, ))
#
#
#
#     #print(loaded_model.predict(np.swapaxes(np.array(data), 0, 1)))
#     #df.to_csv(f, header=False)
#     #dataset = pd.read_csv("output.csv")
#     # adding column labels
#     #dataset.columns=['Oxy1','DeOxy1','Oxy2','DeOxy2','Oxy3','DeOxy3', 'Oxy4','DeOxy4','Oxy5','DeOxy5','Oxy6','DeOxy6','Oxy7','DeOxy7','Oxy8','DeOxy8']
#
#     #print("Predicting the value")
#     #print(loaded_model.predict(rows))
#
#     # printing all the channel values
#     print("oxy1")
#     print (row[0])
#     print("deoxy1")
#     print (row[1])
#     print("oxy2")
#     print (row[2])
#     print("deoxy2")
#     print (row[3])
#     print("oxy3")
#     print (row[4])
#     print("deoxy3")
#     print (row[5])
#     print("oxy4")
#     print (row[6])
#     print("deoxy4")
#     print (row[7])
#     print("oxy5")
#     print (row[8])
#     print("deoxy5")
#     print (row[9])
#     print("oxy6")
#     print (row[10])
#     print("deoxy6")
#     print (row[11])
#     print("oxy7")
#     print (row[12])
#     print("deoxy7")
#     print (row[13])
#     print("oxy8")
#     print (row[14])
#     print("deoxy8")
#     print (row[15])
#
# db.close()

# Clean up
f.close()
cur.close()
#connection.close()
