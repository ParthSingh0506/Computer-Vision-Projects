import cv2

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

while True :
    #Capturing Image Frame By Frame
    ret,frame = video_capture.read()

    image_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    detections = face_detector.detectMultiScale(image_gray,minSize=(100,100))

    #Drawing Rectangle on face
    for (x,y,w,h) in detections :
        cv2.rectangle(frame,(x,y) , (x+w,y+h) , (0,255,255) ,2)

    #Displaying the result
    cv2.imshow('Video',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


#When everything is done release the memory

video_capture.release()
cv2.destroyAllWindows()