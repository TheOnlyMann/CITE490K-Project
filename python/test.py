import cv2

img = cv2.imread("images/modworkshop.png")
print(img[0][0])
print(img)
print(img.shape)#array
#cv2.imshow("MWS",img)