import numpy as np
from midiutil.MidiFile import MIDIFile

data = [[0, 1, 2, 4, 2, 4, 6, 7, 8, 8, 3, 5, 3, 2, 1, 4]]

def createMidi(data):
    MyMIDI = MIDIFile(16)
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


    for i, pitch in enumerate(data[0]):
        #Now add the note.
        MyMIDI.addNote(track,channel,pitch*30,time + i,duration,volume)


    #And write it to disk.
    binfile = open("output.mid", 'wb')
    MyMIDI.writeFile(binfile)
    binfile.close()

createMidi(data)
#There are several additional event types that can be added and there are various options available for creating the MIDIFile object, but the above is sufficient to begin using the library and creating note sequences.
