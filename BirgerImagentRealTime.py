# This python code is use to read the data from COM port. Install pyserial library
# It is connected to the mysql database and if you want to use it on your computer,
# change the host id, port id, user name, passwd, and db name
# The data is save in the "REALTIME1" table
# Originally by Enhao, modified by Nick Chen, then by Sam Hincks

# This code has been modified to also accept input from a CMS50D+ pulse
# oximeter.  It stores the data in the "REALTIME" table.  The format is listed
# below, with an optional timestamp (removable for doing ML analysis).

#http://www.silabs.com/products/mcu/Pages/USBtoUARTBridgeVCPDrivers.aspx

# Byte 1: ?
# Byte 2: Wave form Y-Axis
# Byte 3: ?
# Byte 4: PRbpm
# Byte 5: SpO2


# To get started
# In boxy, press the button to capture

# Type
# python samImagentOld.py 0
# It needs to catch the stream so keep typing it until it stars capturing
# You will get an error like this
#File "samImagentRealTime.py", line 140, in readFromImagent
#    (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",(float(values[2]),float(values[3]),float(values[4]),float(values[5]),float(values[6]),float(values[7]),float(values[8]),float(values[9]),float(values[10]),float(values[11]),float(values[12]),float(values[13]),float(values[14]),float(values[15]),float(values[16]),float(values[17])))
#IndexError: list index out of range

# Don't worry about it, keep typing python samImagentOld.py 0


import sys
if not sys.version_info[0] == 3:
    print("Error: Please use python 3.x.")
    sys.exit(1)

import serial
import pymysql
import datetime
import numpy as np
import pandas as pd
from random import randint
from time import sleep
from keras.models import model_from_json
from midiutil.MidiFile import MIDIFile


#DEVICE = 'CMS50D'
#DEVICE = 'fNIRS'
DEVICE = 'Imagent'
#DEVICE = 'Fake'

"""
Time format info
%y: Year
%m: Month
%d: Day of the month
%H: Hour (24H)
%I: Hour (12H)
%M: Minute
%S: Second
%f: Microsecond

%x: MM/DD/(YY)YY
%X: HH:MM:SS
"""

ADDTIMESTAMP = True
TIMEFORMAT = "%X.%f"

if DEVICE == 'CMS50D':
    ser = serial.Serial(
        port='/dev/tty.SLAB_USBtoUART',\
        baudrate=19200,\
        parity=serial.PARITY_NONE,\
        stopbits=serial.STOPBITS_ONE,\
        bytesize=serial.EIGHTBITS,\
            timeout=None)

#serialport = serial('COM1','InputBufferSize',2048,'BaudRate',57600, ...
    #'StopBits',1,'Terminator','LF','Parity','none','FlowControl','none', ...
    #'Timeout',2);

elif DEVICE == 'Imagent':
    ser = serial.Serial(
        #port='COM1',\
        # new code for Birgers computer
        port ='/dev/tty.usbserial',\
        # old code from Sam
        #port='/dev/cu.usbserial',\
        baudrate=57600,\
        parity=serial.PARITY_NONE,\
        # commented out old settings
        #parity=serial.PARITY_NONE,\
        stopbits=serial.STOPBITS_ONE,\
        # commented out old settings
        #stopbits=serial.STOPBITS_ONE,\
        bytesize=serial.EIGHTBITS,\
        timeout=2)

elif DEVICE == 'fNIRS':
    ser = serial.Serial(
        port='/dev/tty.uart-79FF427A4D083033',\
        baudrate=9600,\
        parity=serial.PARITY_NONE,\
        stopbits=serial.STOPBITS_ONE,\
        bytesize=serial.EIGHTBITS,\
            timeout=0)
else:
    ser = "Fake"




def readToCsv(data, classifier):
    # Opening a csv file for reading
    f = open('testing.csv', 'w')

    #rows = [[row[0]], [row[1]], [row[2]], [row[3]], [row[4]], [row[5]], [row[6]], [row[7]], [row[8]], [row[9]], [row[10]], [row[11]], [row[12]], [row[13]], [row[14]], [row[15]]]

    # Add a classifier as the last row as a number (preferably 0 or 1)

    # Change the dimensions of the input so that it becomes 16 rows FNIRS data (X) and classifier (Y)
    x = data
    y = classifier

    # append y to end of x
    data = np.append(data, y)


    df = pd.DataFrame(data)
    df.to_csv(f, header=False)
    #dataset = pd.read_csv("output.csv")
    # adding column labels
    #dataset.columns=['Oxy1','DeOxy1','Oxy2','DeOxy2','Oxy3','DeOxy3', 'Oxy4','DeOxy4','Oxy5','DeOxy5','Oxy6','DeOxy6','Oxy7','DeOxy7','Oxy8','DeOxy8']



def trainModel(X, Y):
    # The input for the X value should be 16 channels data for FNIRS
    # The input for the Y value should be a classifier of either 0-1
    # Training the model
    model = Sequential()
    model.add(BatchNormalization(input_shape=(16, )))
    model.add(Dropout(0.5))
    model.add(Dense(100, init="normal", activation='relu'))
    model.add(Dense(100, init="normal", activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, init="normal", activation='relu'))
    model.add(Dense(100, init="normal", activation='relu'))
    model.add(Dense(2, init="normal", activation='softmax'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    history = model.fit(X, Y, validation_split=0.3, nb_epoch=300, batch_size=50, verbose=1)

    # Saving the model
    # serialize model to JSON
    model_json = model.to_json()
    with open("trained.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("trained.h5")
    print("Saved model to disk")



#data = [[0, 1, 2, 4, 2, 4, 6, 7, 8, 8, 3, 5, 3, 2, 1, 4]]

def createMidi(musicData):
    MyMIDI = MIDIFile(1)
    #Tracks are numbered from zero. Times are measured in beats.
    track = 0
    time = 0
    channel = 0
    time = 0
    volume = 100
    duration = 1
    #Add track name and tempo.
    MyMIDI.addTrackName(track,time,"Sample Track")
    MyMIDI.addTempo(track,time,120)

    musicData = np.reshape(musicData, (1, 16))

    for value in musicData:
        print (musicData.shape)
        print (len(value))
        j = 0
        while j < 16:
                #Now add the note.
                print("the value of j is")
                print(j)
                track = j
                channel = j
                pitch = int(value[j])*10
                time = time + j
                MyMIDI.addNote(track,channel, pitch, time, duration, volume)
                j = j + 1

    #And write it to disk.
    binfile = open("output.mid", 'wb')
    MyMIDI.writeFile(binfile)
    binfile.close()

#createMidi(data)



def predict(data):

    # right now the data is getting 16 values, you can store it into a larger matrix 16x60 perhaps and
    # make a prediction every two seconds based on average

    # then we need to change training wiht an appropiate size of input

    #to train we need a task that is 0-5




    # many variables
    # 5 states
    # cognitive load, really low, low, medium, high, very high
    # 5 label classifier or 10

    # load json and create model
    json_file = open('eyes.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("eyes.h5")
    #print("Loaded model from disk")
    # predicting the model
    print("Predicting the value")
    prediction = loaded_model.predict(np.swapaxes(np.array(data), 0, 1))
    #print(loaded_model.predict(np.swapaxes(np.array(data), 0, 1)))
    print(prediction)
    print("prediction [0]")
    print(np.floor(prediction[0]))
    print("prediction [0][0]")
    print(np.floor(prediction[0][0]))

    # this can definitely be improved for making better predictions

    if (np.floor(prediction[0][0]) == 0.0):
        print("open")
    elif (np.floor(prediction[0][0]) == 1.0):
        print("closed")
    else:
        print("error")
#
# def readFromFake():
#     print("Printing Fake");
#     it =0;
#     while True:
#         conn = pymysql.connect(host='127.0.0.1', port=3306,
#                     user='root', db='newttt')
#         cur=conn.cursor()
#
#         it = it +1
#         max =it
#         if it > 100:
#         	max = it
#         #Insert the data to the Table REALTIME
#         cur.execute("""INSERT INTO REALTIME1(A1HBO,A1HB,A2HBO,A2HB,A3HBO,A3HB,A4HBO,A4HB,B1HBO,B1HB,B2HBO,B2HB,B3HBO,B3HB,B4HBO,B4HB) VALUES
#           (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",(randint(1,max),randint(1,max),randint(1,3),randint(1,3),randint(1,3),randint(1,3),randint(1,3),randint(1,3),randint(1,3),randint(1,3),randint(1,3),randint(1,3),randint(1,3),randint(1,3),randint(1,3),randint(1,3)))
#
#
#         conn.commit()
#         cur.close()
#         conn.close()
#         sleep(0.5)

# code for running real time system
def readFromImagent():
    count=1
    output = str('')
    s = ""
    print("About to start reading, and spamming if we see values");

    errorcount = 0
    predictcounter = 0

    printed = False
    while True:
        for line in ser.read():
            # converting from integer to ASCII
            cha = chr(line)
            # converting to string
            s = s + cha
            if line == 10:  # if line is equal to ASCII character \n
                # 13 is the \r and we skip it because it isn't a part of data
                ser.read() # read away the 13
                # now s is the entire line. Do something with it
                #print (s)
                values = s.split()
                if len(values) != 18:
                    s=""
                    continue
                #print (len(values))
                if (not printed):
                	print(values)
                	print(s)
                	print("---")
                	printed = True
                s = ""

                #conn = pymysql.connect(host='127.0.0.1', port=3306,
                #            user='root', db='newttt')
                #cur=conn.cursor()

                #print(values)
                try:
                    #cur.execute("""INSERT INTO REALTIME1(A1HBO,A1HB,A2HBO,A2HB,A3HBO,A3HB,A4HBO,A4HB,B1HBO,B1HB,B2HBO,B2HB,B3HBO,B3HB,B4HBO,B4HB) VALUES
                    #(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",(float(values[2]),float(values[3]),float(values[4]),float(values[5]),float(values[6]),float(values[7]),float(values[8]),float(values[9]),float(values[10]),float(values[11]),float(values[12]),float(values[13]),float(values[14]),float(values[15]),float(values[16]),float(values[17])))

                    data = [(float(values[2]),float(values[3]),float(values[4]),float(values[5]),float(values[6]),float(values[7]),float(values[8]),float(values[9]),float(values[10]),float(values[11]),float(values[12]),float(values[13]),float(values[14]),float(values[15]),float(values[16]),float(values[17]))]
                except ValueError:
                    print("caught the error")
                    errorcount = errorcount + 1
                    print("error count")
                    print(errorcount)
                    continue
                # interviewing between oxy and deoxy channel values to have same format as Tomokis data
                #oxydeoxy = ([data[0], data[4], data[1], data[5], data[2], data[6], data[3], data[7], data[8], data[12], data[9], data[13], data[10], data[14], data[11], data[15])

                # setting data to oxy
                #data = oxydeoxy

                #print(data.shape)
                data = np.swapaxes(data,0,1)
                #print(data.shape)
                #print("Printing the value of data")
                print(data)


                predictcounter = predictcounter + 1
                if predictcounter %60 == 0:
                    predict(data)






                #classifier = 0
                #readToCsv(data, classifier)

                #createMidi(data)


                #conn.commit()
                #cur.close()
                #conn.close()
                #output = str('$')
            count = count +1

#
# def readFromfNIRS():
#     count=1
#     output = str('')
#
#     while True:
#         for line in ser.read(2048):
#             cha = chr(line)
#             if cha != '$':
#                 output = output + cha
#             else:
#                 if count == 1:
#                     count = count + 1
#                 else:
#                     print(str(count)+str(':') + output)
#                     li = output.split(sep=",", maxsplit=2)
#
#                     #Remove the "$" form the string
#                     channel1 = li[0].replace("$","")
#                     channel2 = li[1]
#                     print(channel1)
#                     print(channel2)
#
#
#     #Connct to the DB newttt
#                     conn = pymysql.connect(host='127.0.0.1', port=3306,
#                             user='root', db='newttt')
#                     cur=conn.cursor()
#
#     #Insert the data to the Table REALTIME
#                     cur.execute("""INSERT INTO REALTIME1(Channel1,Channel2) VALUES
#                       (%s,%s)""",(channel1,channel2))
#                     conn.commit()
#                     cur.close()
#                     conn.close()
#                     output = str('$')
#                     count = count+1


def addChunkToDB(host, port, user, pw, db, chunk):
    for i in range(len(chunk)):
        chunk[i] = str(chunk[i])
    conn = pymysql.connect(host=host, port=port, user=user, db=db)
    cur=conn.cursor()

    #Insert the data to the Table REALTIME
    if ADDTIMESTAMP:
        time = datetime.datetime.now().strftime(TIMEFORMAT)
        cur.execute("""INSERT INTO REALTIME(Uk1,YAxis,Uk2,PRbpm,SpO2,Time) VALUES
          (%s,%s,%s,%s,%s,%s)""",(chunk[0],chunk[1],chunk[2],chunk[3],chunk[4],time))
    else:
        cur.execute("""INSERT INTO REALTIME(Uk1,YAxis,Uk2,PRbpm,SpO2) VALUES
          (%s,%s,%s,%s,%s)""",(chunk[0],chunk[1],chunk[2],chunk[3],chunk[4]))
    conn.commit()
    cur.close()
    conn.close()

def readChunk(chunkSize):
    return [ser.read() for i in range(5)]

def printData(data, trans=True):
    # First chunk is sometimes not the right size (??)
    data.pop(0)
    if trans:
        for set in zip(*data):
            print(set)
            print()
    else:
        print(data)

def readFromCMS50D():
    chunkSize = 5
    aligned = False

    data = []
    oneChunk = []

    i = 0
    while True:
        line = ser.read()
        # The first byte is the only one that is > 127, so align based on that
        if not aligned:
            if line > 127:
                aligned = True
                oneChunk.append(line)
        else:
            oneChunk.append(line) #oneChunk = readChunk(chunkSize)
            i+=1
            if len(oneChunk) == chunkSize: # Chunk reading complete
                print(oneChunk)
                data.append(oneChunk)
                addChunkToDB('127.0.0.1', 3306, 'root', 'fnirs196',
                        'newttt', oneChunk)
                oneChunk = []
                aligned = False

def main():
    print("running the main function")
    if sys.argv[1] =="2":
        DEVICE = "Fake"
    else:
        DEVICE = "Imagent"
        print ("running on imagent")


    conn = pymysql.connect(host='127.0.0.1', port=3306,
            user='root',db='newttt')
    cur=conn.cursor()

    if (DEVICE != "Fake"):
        print("connected to: " + ser.portstr)


    if DEVICE == "CMS50D":
        tableName = "REALTIME"
        cur.execute("DROP TABLE IF EXISTS " + tableName)
        if ADDTIMESTAMP:
            createQuery = "CREATE TABLE " + tableName +\
            " (Uk1 VARCHAR(45), YAxis VARCHAR(45), Uk2 VARCHAR(45), PRbpm VARCHAR(45), SpO2 VARCHAR(45), Time VARCHAR(45))";
        else:
            createQuery = "CREATE TABLE " + tableName +\
            " (Uk1 VARCHAR(45), YAxis VARCHAR(45), Uk2 VARCHAR(45), PRbpm VARCHAR(45), SpO2 VARCHAR(45))";
        cur.execute(createQuery)
        readFromCMS50D()
    elif DEVICE == "fNIRS":
        tableName = "REALTIME1"
        cur.execute("DROP TABLE IF EXISTS " + tableName)
        createQuery = "CREATE TABLE " + tableName +" (Channel1 VARCHAR(45), Channel2 VARCHAR(45))";
        cur.execute(createQuery)
        readFromfNIRS()
    elif DEVICE == "Imagent":
        tableName = "REALTIME1"
        cur.execute("DROP TABLE IF EXISTS " + tableName)
        print("running Imagent")
        createQuery = "CREATE TABLE " + tableName +" (A1HBO VARCHAR(45), A1HB VARCHAR(45), A2HBO VARCHAR(45), A2HB VARCHAR(45), A3HBO VARCHAR(45), A3HB VARCHAR(45), A4HBO VARCHAR(45), A4HB VARCHAR(45),B1HBO VARCHAR(45), B1HB VARCHAR(45), B2HBO VARCHAR(45), B2HB VARCHAR(45), B3HBO VARCHAR(45), B3HB VARCHAR(45), B4HBO VARCHAR(45), B4HB VARCHAR(45))";
        cur.execute(createQuery)
        readFromImagent()
    elif DEVICE == "Fake":
        tableName = "REALTIME1"
        print("Device is fake")
        cur.execute("DROP TABLE IF EXISTS " + tableName)
        createQuery = "CREATE TABLE " + tableName +" (A1HBO VARCHAR(45), A1HB VARCHAR(45), A2HBO VARCHAR(45), A2HB VARCHAR(45), A3HBO VARCHAR(45), A3HB VARCHAR(45), A4HBO VARCHAR(45), A4HB VARCHAR(45),B1HBO VARCHAR(45), B1HB VARCHAR(45), B2HBO VARCHAR(45), B2HB VARCHAR(45), B3HBO VARCHAR(45), B3HB VARCHAR(45), B4HBO VARCHAR(45), B4HB VARCHAR(45))";

        cur.execute(createQuery)
        readFromFake()

    ser.close()

if __name__ == "__main__":
    print("hello main")
    main()
