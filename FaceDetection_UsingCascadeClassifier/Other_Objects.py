import cv2

image = cv2.imread('car.jpg')
image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

car_detector = cv2.CascadeClassifier('cars.xml')
detections = car_detector.detectMultiScale(image_gray,scaleFactor=1.01,minNeighbors=5
                                           ,minSize=(30,30),maxSize=(80,80))

for (x,y,w,h) in detections :
    print(w,h)
    cv2.rectangle(image,(x,y) , (x+w,y+h) , (0,255,0) , 2)

cv2.imshow('',image)
cv2.waitKey()