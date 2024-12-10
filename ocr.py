import easyocr
import numpy as np
import cv2
from segmentator import *

class EasyOcr:

  USELESS_TEXT = ['BR', 'BRASIL', 'Brasil']

  reader = easyocr.Reader(['pt','en'], gpu=False)

  def __init__(self, pre_processor = None) -> None:
    """
    receives a pre processor, this is a function that pre process the image to recognize
    if None is passed it uses the default_pre_processor function
    """
    self.segmentator = PlateSegmentator()

    if pre_processor == None:
      self.pre_processor = self.default_pre_processor
    
  def default_pre_processor(self, image):
    """
    Process a image limiarizing it by a set interval, raising the contrast\n
    and passing a Gaussian filter 
    """
    image = cv2.bilateralFilter(image, d=3, sigmaColor=20, sigmaSpace=15)
    cv2.imshow("bilateral filter", image)
    cv2.waitKey(0)

    image = cv2.inRange(image, np.array([0]), np.array([125]))   # Combine results
    cv2.imshow("limiarizing", image)
    cv2.waitKey(0)
    return image
  
  def recognize_plates(self, plates):
    plates_ocr = []


    for plate in plates:
      processed = plate

      ocr = self.reader.readtext(processed, detail=1)

      for remove_txt in self.USELESS_TEXT:
        for j in ocr:
          rmv = remove_txt.upper()
          y = [x.upper() for x in j if type(x) == str]
          if remove_txt in y:
            ocr.remove(j)

      if len(ocr) < 1:
        return []

      ocr = ocr[0]
      
      plates_ocr.append(ocr)

    return plates_ocr

  def run(self, image):

    original_image = np.copy(image)
    if len(image.shape) == 3:
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    boundaries = self.segmentator.detect_bounding_box(image)
    
    plates = self.segmentator.segment_img(image, boundaries)

    plates = [self.pre_processor(plate) for plate in plates]

    ocr = self.recognize_plates(plates)

    if len(ocr) == 0:
      print("could'n recognize")

    for plate in ocr:
      bounding_box = plate[0]
      text = plate[1]
      plates = np.array(plates)
      plates = cv2.putText(
        original_image, 
        text="Plate: " + str(text), 
        org=[bounding_box[3][0] - 20, bounding_box[3][1] + 20],
        fontFace=cv2.FONT_HERSHEY_COMPLEX,
        fontScale=1,
        color=(0, 0, 0),
        thickness=2,
        lineType=cv2.LINE_AA
        )
      
      cv2.imshow(f"ocr of plate: {text}", plates)
      cv2.waitKey(0)
      




