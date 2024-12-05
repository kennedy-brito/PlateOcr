import cv2
from ocr import EasyOcr
from segmentator import *
import numpy as np
imgs_path =  ['MSF6757.jpg','KVF9175.jpg' , 'FLT6A14.jpg', 'image.png','AUQ2A27.jpg']
app = EasyOcr()


for img_path in imgs_path:
  img = cv2.imread(img_path)
  app.run(img)