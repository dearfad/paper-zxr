import cv2
image_path = './img/24h-10x-center.tif'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Image', image)
cv2.imshow('Gray', gray)
cv2.waitKey(0)