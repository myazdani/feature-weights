import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from PIL import Image
from skimage.transform import rotate
import cv2

class ObjectRotateDetector(BaseEstimator, ClassifierMixin):
  def __init__(self,  cascadePath = "/Applications/OpenCV/opencv-2.4.8/data/haarcascades/haarcascade_frontalface_alt.xml", scaleFactor = 1.1, minNeighbors = 3, maxAngle = 10, angleStepSize = 2):
    self.cascadePath = cascadePath
    self.scaleFactor = scaleFactor
    self.minNeighbors = minNeighbors
    self.maxAngle = maxAngle
    self.angleStepSize = angleStepSize
    self.cascade_path = cascadePath
    #self.objectClassifier = cv2.CascadeClassifier(cascadePath)

  def _equalizedGray(self, image):
    im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(im)
  
  def _getImage(self, image_path):
    im = cv2.imread(image_path)
    gray = self._equalizedGray(im)
    return gray

  def fit(self, X, y = None):
    self.objectClassifier = cv2.CascadeClassifier(self.cascade_path)
    assert (type(X) == list), "X must be list"
    return self

  def _rotator(self, gray):
    for angle in range(0, self.maxAngle, self.angleStepSize):
      gray_rotated = np.array(rotate(gray, angle, preserve_range=True), dtype = "uint8")
      found_objects = self.objectClassifier.detectMultiScale(gray_rotated, scaleFactor = self.scaleFactor, minNeighbors = self.minNeighbors)
      ## if we find 1 or more objects of interest, 
      ## break from angle rotator loop
      if len(found_objects) > 0: 
        break
    return found_objects


  def predict(self, X, y = None):
    print "using", self.cascadePath
    res = []
    for x in X:
      gray = self._getImage(x)
      found_objects_orig = self._rotator(gray)
      gray_flipped = np.array(Image.fromarray(gray).transpose(Image.FLIP_LEFT_RIGHT))
      found_objects_flipped = self._rotator(gray_flipped)
      if len(found_objects_orig) > len(found_objects_flipped):
        found_objects = found_objects_orig
      else:
        found_objects = found_objects_flipped
      
      ## append last saved found_object list to res 
      res.append(np.sign(len(found_objects)))
    res_arr = np.array(res)
    return res_arr.ravel()
