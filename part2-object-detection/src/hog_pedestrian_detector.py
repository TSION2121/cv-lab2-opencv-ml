import cv2

# Initialize HOG + SVM
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Load image
img = cv2.imread('../pedestrians.jpg')
img = cv2.resize(img, (640, 480))

# Detect pedestrians
(rects, weights) = hog.detectMultiScale(img, winStride=(8,8), padding=(8,8), scale=1.05)

# Draw rectangles
for (x, y, w, h) in rects:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('Pedestrian Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
