# Smart-Watch-Design
In this project, to design a smart wristwatch that has the capability of geographic positioning as well as measuring the heart rate of the person under care, as well as processing the data related to the heart rate sensor and processing the deep recursive network on this data and detecting the current state of the person, i.e. "The person has fallen" or "The person is trapped" or "The person is alive or dead" and finally "The person is injured". To achieve this, the Sisfall dataset and also the detection of the falling situation with the help of Recurrent Deep Neural Network (RNN) and finally putting the trained model on the esp32-wroom32 microcontroller and processing the real data received from the pulse sensor and sending the output result to the server environment which in this project is from Google Sheet for Display and storage of data related to the location, as well as data related to the heart rate sensor and data related to the output of the trained model (one of the 5 types of activities related to daily activities and falling), are used In this project, in order to detect whether a person is trapped or alive or dead or injured, by using the heart rate sensor data that can be seen in the Google Sheet environment, these situations can be In order to diagnose the condition of a person being wounded, this condition can be understood from the decrease in the heart rate, or in a period of time when the location of the person does not change and the heart rate is increasing, this possibility can be given. that the person is confined in this location, and if the heart rate becomes zero, it means that the person has cardiac asystole and is dead. In this project, low-power modules are used, which increases The life of the battery and the power supply of the device is consumed. Therefore, in this project, 3.7 V 1000 mAh battery modules are used to power the device, Ublox Neo6M GPS module is used to determine the geographic location, and the pulse sensor module is used as above. It was mentioned, it was used to determine the heart rate of the person under care.
The link related to this project is given below and it should be mentioned that Python language was used to design the recurrent deep neural network (RNN) and Arduino programming language was used to code the esp32-wroom32 microcontroller, in the not too distant future it will be possible to detect The four mentioned situations are possible with the help of artificial intelligence.

Link to Google Drive, description of the smart wristband design project:
https://drive.google.com/file/d/1uN2HTS5Wt-oqSwqVQp-O_pPVzVAf4x2C/view?usp=drive_link

Google Drive link of the project and all class activities:
https://drive.google.com/drive/folders/1t60cGLn013Ep5F4Ft_9nn4Ba7i-FDGYs?usp=drive_link
