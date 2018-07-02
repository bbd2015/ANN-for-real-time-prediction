def readFromImagent():
    count=1
    output = str('')
    s = ""
    print("About to start reading, and spamming if we see values");

    errorcount = 0

    printed = False
    while True:
        for line in ser.read():
            #if(line != 18)
                #ser.read(18-line)
            print('line', line)
            #print(line)
            # converting from integer to ASCII
            cha = chr(line)
            print(cha)
            # converting to string
            s = s + cha
            if line == 10:
                # 13 is the \r and we skip it because it isn't a part of data
                print ('before', chr(line))
                leon=ser.read() # read away the 13
                print ('leon', leon)
                # now s is the entire line. Do something with it

                values = s.split()
                print (len(values))
                if (not printed):
                	print(values)
                	print(s)
                	print("---")
                	printed = True
                s = ""

                conn = pymysql.connect(host='127.0.0.1', port=3306,
                            user='root', db='newttt')
                cur=conn.cursor()

                print(values)
                try:
                    cur.execute("""INSERT INTO REALTIME1(A1HBO,A1HB,A2HBO,A2HB,A3HBO,A3HB,A4HBO,A4HB,B1HBO,B1HB,B2HBO,B2HB,B3HBO,B3HB,B4HBO,B4HB) VALUES
                    (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",(float(values[2]),float(values[3]),float(values[4]),float(values[5]),float(values[6]),float(values[7]),float(values[8]),float(values[9]),float(values[10]),float(values[11]),float(values[12]),float(values[13]),float(values[14]),float(values[15]),float(values[16]),float(values[17])))
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

                data = np.swapaxes(data,0,1)
                print("Printing the value of data")
                print(data)
                predict(data)

                readToCsv(data)

                #createMidi(data)


                conn.commit()
                cur.close()
                conn.close()
                output = str('$')
            count = count +1
