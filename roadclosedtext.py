import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract

src_path_script = "C:/Users/Emily/Documents/Python Scripts/img/"
src_path_pic = "C:/Users/Emily/Downloads/road_closed_sign-20180108T142629Z-001/road_closed_sign/"

#Read img with opencv and convert to gray
img = cv2.imread(src_path_pic + "road_closed0.png"
              , cv2.IMREAD_GRAYSCALE)

#Apply dilation and erosion to remove some noise
kernel = np.ones((1,1), np.uint8)
img = cv2.dilate(img, kernel, iterations = 1)
img = cv2.erode(img, kernel, iterations = 1)


cv2.imwrite(src_path_script + "removed_noise.png",img)


#Apply threshold to get img with only black and white
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY, 11,2)
cv2.imwrite(src_path_script + "thres.png",img)


#recognize text with tesseract for python
result = pytesseract.image_to_string(cv2.imread(src_path_script + "thres.png"))


#Recognize text with tesseract for python
print(result)


cv2.imshow('cvimage',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

##plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
##plt.show()


