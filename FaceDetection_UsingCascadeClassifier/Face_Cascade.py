import cv2

#Loading the image
image = cv2.imread('people1.jpg')

#Reshaping the image#
image = cv2.resize(image,(1280,720))

#Converting the image into gray
image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#Detecting faces

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
detection = face_detector.detectMultiScale(image_gray,scaleFactor=1.3,minSize=(30,30),maxSize=(100,100))

for (x,y,w,h) in detection :
  cv2.rectangle(image,(x , y) , (x + w, y + h),(0,255,0),2)

#Eye detector

eye_detector = cv2.CascadeClassifier('haarcascade_eye.xml')
detections = eye_detector.detectMultiScale(image_gray,scaleFactor=1.1,maxSize=(50,50))

for (x,y,w,h) in detections:
    print(w,h)
    cv2.rectangle(image,(x,y) ,(x + w , y + h), (0,0,255) , 2)

cv2.imshow('',image)
cv2.waitKey()
