from segmentator import *
import matplotlib.pyplot as plt
import cv2
import numpy as np
img_path =  "image.png"

segmentator = PlateSegmentator()

img = cv2.imread(img_path)


bounding_boxes = segmentator.detect_bounding_box(img)

segmented_imgs = segmentator.segment_img(img, bounding_boxes)

for i in segmented_imgs:
  
  i = cv2.inRange(i, np.array([0]), np.array([100]))
  while True:
    cv2.imshow(
      "image ",
      i)
    
    if cv2.waitKey(1) & 0xFF == ord("q"): #quit after pressing the key 'q'
      break
