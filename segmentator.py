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
    
    plates_position = self.classifier.detectMultiScale(img, 1.1, 5, minSize=(40, 40))

    
    return plates_position
  
  def segment_img(self, img, plates_position):

    imgs = []

    for (x, y, width, height) in plates_position:
      segmented_img = img[y:y+height, x:x+width]
      imgs.append(segmented_img)
    cv2.imshow("plate", imgs[0])
    cv2.waitKey(0)
    return imgs
  
  def run(self, img):
    """
    Segments the image plates
    """
    bounding_boxes = self.detect_bounding_box(img) 

    return self.segment_img(img, bounding_boxes)