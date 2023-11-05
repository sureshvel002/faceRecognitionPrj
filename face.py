import cv2

#trained dataset
trainedDataset = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#read a img
img=cv2.imread('images/men-over-26.jpg')

#image into greyscale
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces= trainedDataset.detectMultiScale(gray)
print(faces)
for x, y, w, h in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
cv2.imshow('squad',img)
# cv2.imshow('gray',gray)
cv2.waitKey()