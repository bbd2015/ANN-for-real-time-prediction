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
# Type
# python samImagentOld.py 0

import sys
if not sys.version_info[0] == 3:
    print("Error: Please use python 3.x.")
    sys.exit(1)

import serial
import pymysql
import datetime
from random import randint
from time import sleep

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
        port='/dev/tty.usbserial',\
        baudrate=57600,\
        parity=serial.PARITY_NONE,\
        stopbits=serial.STOPBITS_ONE,\
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




def readFromFake():
    print("Printing Fake");
    it =0;
    while True:
        conn = pymysql.connect(host='127.0.0.1', port=3306,
                    user='root', db='newttt')
        cur=conn.cursor()

        it = it +1
        max =it
        if it > 100:
        	max = it
        #Insert the data to the Table REALTIME
        cur.execute("""INSERT INTO REALTIME1(A1HBO,A1HB,A2HBO,A2HB,A3HBO,A3HB,A4HBO,A4HB,B1HBO,B1HB,B2HBO,B2HB,B3HBO,B3HB,B4HBO,B4HB) VALUES
          (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",(randint(1,max),randint(1,max),randint(1,3),randint(1,3),randint(1,3),randint(1,3),randint(1,3),randint(1,3),randint(1,3),randint(1,3),randint(1,3),randint(1,3),randint(1,3),randint(1,3),randint(1,3),randint(1,3)))


        conn.commit()
        cur.close()
        conn.close()
        sleep(0.5)

def readFromImagent():
    count=1
    output = str('')
    s = ""
    print("About to start reading, and spamming if we see values");
    printed = False
    while True:
        for line in ser.read():
            cha = chr(line)
            s = s + cha
            if line == 10:
                ser.read() # read away the 13
                # now s is the entire line. Do something with it

                values = s.split()
                if (not printed):
                	print(values)
                	print(s)
                	print("---")
                	printed = True
                s = ""

                conn = pymysql.connect(host='127.0.0.1', port=3306,
                            user='root', db='newttt')
                cur=conn.cursor()

                #Insert the data to the Table REALTIME
                cur.execute("""INSERT INTO REALTIME1(A1HBO,A1HB,A2HBO,A2HB,A3HBO,A3HB,A4HBO,A4HB,B1HBO,B1HB,B2HBO,B2HB,B3HBO,B3HB,B4HBO,B4HB) VALUES
                  (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",(float(values[2]),float(values[3]),float(values[4]),float(values[5]),float(values[6]),float(values[7]),float(values[8]),float(values[9]),float(values[10]),float(values[11]),float(values[12]),float(values[13]),float(values[14]),float(values[15]),float(values[16]),float(values[17])))



                conn.commit()
                cur.close()
                conn.close()
                output = str('$')
            count = count +1


def readFromfNIRS():
    count=1
    output = str('')

    while True:
        for line in ser.read(2048):
            cha = chr(line)
            if cha != '$':
                output = output + cha
            else:
                if count == 1:
                    count = count + 1
                else:
                    print(str(count)+str(':') + output)
                    li = output.split(sep=",", maxsplit=2)

                    #Remove the "$" form the string
                    channel1 = li[0].replace("$","")
                    channel2 = li[1]
                    print(channel1)
                    print(channel2)


    #Connct to the DB newttt
                    conn = pymysql.connect(host='127.0.0.1', port=3306,
                            user='root', db='newttt')
                    cur=conn.cursor()

    #Insert the data to the Table REALTIME
                    cur.execute("""INSERT INTO REALTIME1(Channel1,Channel2) VALUES
                      (%s,%s)""",(channel1,channel2))
                    conn.commit()
                    cur.close()
                    conn.close()
                    output = str('$')
                    count = count+1


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
    if sys.argv[1] =="2":
        DEVICE = "Fake"
    else:
        DEVICE = "Imagent"


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
    main()
