
import sqlite3
import pandas as pd

# Open the file
f = open('output.csv', 'w')
# Create a connection and get a cursor
connection = sqlite3.connect('REALTIME1.sql')
cursor = connection.cursor()
# Execute the query
cursor.execute('select * from REALTIME1')
# Get data in batches
while True:
    # Read the data
    df = pd.DataFrame(cursor.fetchmany(1000))
    # We are done if there are no data
    if len(df) == 0:
        break 
    # Lets write to the file
    else:
        df.to_csv(f, header=False)

# Clean up
f.close()
cursor.close()
connection.close()
