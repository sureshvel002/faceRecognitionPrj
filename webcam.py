import cv2
trainedDataset = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video=cv2.VideoCapture(0)
while True:
    success,frame=video.read()
    if success==True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces= trainedDataset.detectMultiScale(gray)
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imshow('squad', frame)
        close=cv2.waitKey(1)
        if close & 0xFF == ord('q'):
            break
    else:
        print("video complete")
        break
cv2.destroyAllWindows()