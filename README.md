# Cognitive-Load
This is a repository for analysing FNIRS data for cognitive load. It uses python 3 with Tensorflow 
and Keras for building the machine learning models.
To get started you need to download Anaconda.
https://www.continuum.io/downloads
Once you clone down the repository you can then go to the folder your files are stored in an type jupyter notebook.
This will open up an interactive environment of python files. 
The file called Cognitive Load.ipynb is the main file.
This file is used for training machine learning models on cognitive load data.

## Long term goals
We are working trying to improve the models by training them on data regarding N-back and Sorting task.
The next step is implementing these stored neural network models as a trained filter for our data. 

## Filter fro predicting real time data
The file samImagentRealtime.py is used for catching streaming FNIRS data from Boxy. To get the real time data on your computer you need to setup a few things. Here follows information on how to make your computer ready for getting real time data, (this is for a mac computer)

##Installation for using a real time filter on Mac computer
 
Get JDK 8   
http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html
Get Netbeans 
https://netbeans.org/downloads/
From Netbeans, pull from Github
https://github.com/samhincks/Neuracle
Install mySQL 2
Download homebrew
 ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
 brew install mysql 
https://dev.mysql.com/doc/refman/4.1/en/macosx-installation.html 
Install xcode (command line dev tools work as well)  
brew install python3 
“pip3 install pyserial”
“pip3 install pymysql”
to test mySQ:.  Open samImagentRealtime.py
un-comment “DEVICE = fake”	
      g.	run - python3 samImagentrealtime.py 	
      h.	install mysql workbench https://dev.mysql.com/downloads/workbench/ 
 
Mysql.start -- don’t forget this
1. sudo /Library/StartupItems/MySQLCOM/MySQLCOM start
2. mysql -u root
3. show databases;
mysql> CREATE DATABASE newttt;
use database newttt
		 
Install driver
http://www.trendnet.com/support/supportdetail.asp?prod=265_TU-S9#tabs-solution01
Verify successful installation by going to System Information -> USB 
make sure USB-Serial Controller D is present
 
fNIRS calibration
Turn on fNIRS machine (in the back)
Turn on BOXY machine, login with 
HCI-Jacob01\user
password: Jacob01
Open up BOXY, and follow a subset of the instructions in the Instructions for BOXY manual, up to Measuring on Subject
Then put fNIRS on your own forehead, making sure it is fastened tightly 
 
Running the software
Plug usb cable into machine
From command line, write samImagentRealtime.py . Make sure that DEVICE = ‘Imagent’
