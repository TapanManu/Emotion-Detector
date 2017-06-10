import cv2
import numpy as np
import time
import glob
import os
import Update_Model

video_capture = cv2.VideoCapture(0)
facecascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
fishface = cv2.createFisherFaceRecognizer()
fishface.load("trained_emoclassifier.xml")

facedict = {}
emotions = ["angry", "happy", "sad", "fear"]

def crop_face(clahe_image, face):
    for (x, y, w, h) in face:
        faceslice = clahe_image[y:y+h, x:x+w]
        faceslice = cv2.resize(faceslice, (350, 350))
    facedict["face%s" %(len(facedict)+1)] = faceslice
    return faceslice

def recognize_emotion():
    predictions = []
    confidence = []
    
    for x in facedict.keys():
        pred, conf = fishface.predict(facedict[x])
        cv2.imwrite("images\\%s.jpg" %x, facedict[x])
        predictions.append(pred)
        confidence.append(conf)
    print("I think you're %s" %emotions[max(set(predictions), key=predictions.count)])

  
while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)
    face = facecascade.detectMultiScale(clahe_image, scaleFactor=1.1, minNeighbors=15, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in face: 
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2) 
    if len(face) == 1: 
        faceslice = crop_face(clahe_image, face)
    else:
        print("no/multiple faces detected, passing over frame")

    if len(facedict) == 10:
        recognize_emotion()

    cv2.imshow("webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break