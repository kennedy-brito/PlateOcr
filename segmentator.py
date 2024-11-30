import cv2
import numpy

class PlateSegmentator:

  classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

  def __init__(self) -> None:
    pass



  def detect_bounding_box(self, img):
    """
    Detect plates positions in an img\n
    params:\n
      \timg: the image
    return:\n
      \tplates_position: a list of positions in plates 
      \t\ta position is an array with (x, y, width, height)
    """
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    plates_position = self.classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))

    
    return plates_position
  
  def segment_img(self, img, plates_position):

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgs = []

    for (x, y, width, height) in plates_position:
      segmented_img = gray_img[y:y+height, x:x+width]
      imgs.append(segmented_img)
    
    return imgs