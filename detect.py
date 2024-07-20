import cv2
import numpy as np
import os
import sqlite3
import serial,time

facedetect=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
vid=cv2.VideoCapture(1)

ArduinoSerial=serial.Serial('COM4',9600,timeout=0.1)
time.sleep(1)

recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizer/trainingdata.yml")

def getprofile(id):
    conn=sqlite3.connect("sqlite (1).db")
    cursor=conn.execute("SELECT * FROM STUDENTS WHERE id=?",(id,))
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile

while True:
    ret, frame = vid.read()
    
    flip=cv2.flip(frame, 1)  # Read a frame from the camera
    col=cv2.cvtColor(flip,cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(col,1.3,5)

    


    

    for(x,y,w,h) in faces:
        string='X{0:d}Y{1:d}'.format((x+w//2),(y+h//2))
        print(string)
        ArduinoSerial.write(string.encode('utf-8'))
        cv2.circle(frame,(x+w//2,y+h//2),2,(0,255,0),2)
        cv2.rectangle(flip,(x,y),(x+w,y+h),(0,255,0),2)
        id,conf=recognizer.predict(col[y:y+h,x:x+w])
        profile=getprofile(id)
        print(profile)
        if(profile!=None):
            cv2.putText(flip,"Name:"+str(profile[1]),(x,y+h+20),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,127),2)
        
        
            

    
    cv2.imshow("Face",flip)
    if(cv2.waitKey(1)==ord('q')):
        break



vid.release()
cv2.destroyAllWindows()

