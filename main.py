import cv2
import numpy as np
import sqlite3
import serial,time

face=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
vid=cv2.VideoCapture(1)

ArduinoSerial=serial.Serial('COM4',9600,timeout=0.1)
time.sleep(1)


def insertorupdate(Id, Name):
    conn=sqlite3.connect("sqlite (1).db")
    cmd="SELECT * FROM STUDENTS WHERE ID="+str(Id)
    cursor=conn.execute(cmd)
    isRecordExist=0
    for row in cursor:
        isRecordExist=1
    if(isRecordExist==1):
        conn.execute("UPDATE STUDENTS SET Name=? WHERE Id=?",(Name,Id,))
        
    else:
        conn.execute("INSERT INTO STUDENTS(Id, Name) values(?,?)",(Id,Name))
    
    conn.commit()
    conn.close()

Id=input("enter user id:")
Name=input("enter user name:")


insertorupdate(Id,Name)

sampleNum=0
while True:
    ret, frame = vid.read()
    
    flip=cv2.flip(frame, 1)  # Read a frame from the camera
    
    facedet=face.detectMultiScale(flip,
                                  scaleFactor=1.1,
                                  minNeighbors=5,
                                  minSize=(30,30),
                                  flags=cv2.CASCADE_SCALE_IMAGE
                                  )
    for(x,y,w,h) in facedet:
        string='X{0:d}Y{1:d}'.format((x+w//2),(y+h//2))
        print(string)
        ArduinoSerial.write(string.encode('utf-8'))
        cv2.circle(frame,(x+w//2,y+h//2),2,(0,255,0),2)
        sampleNum=sampleNum+1
        cv2.imwrite("C:/robotface/dataset/user."+str(Id)+"."+str(sampleNum)+".jpg",flip[y:y+h,x:x+w])
        cv2.rectangle(flip,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.waitKey(100)
    cv2.imshow('frame', flip) 
    cv2.waitKey(1)
    if (sampleNum>50):
        break # Display the frame

    

    
    
vid.release()
cv2.destroyAllWindows()





    

    
    


