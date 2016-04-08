import numpy as np
from skimage.feature import hog
from sklearn.base import BaseEstimator, TransformerMixin
from ObjectDetector import ObjectRotateDetector
import cv2

'''Transformer classes defined here:
  HogTransformer
  HueHistTransformer
  DimTransformer
  BWTransformer
'''


class HogTransformer(BaseEstimator, TransformerMixin):
  '''Compute the HOG (from SKIMAGE) of list of image arrays

  Parameters
  ----------
  ors: orientations 
  ppc: pixels_per_cell
  cpb: cells_per_block

  Attributes
  ----------
  None

  Examples
  --------
  from feature_util import HogTransformer
  from feature_util import BWTransformer
  imbw = BWTransformer().transform(image_path)
  hog_f = HogTransformer().transform(imbw)
  '''
  def __init__(self, ors=8, ppc=(16, 16), cpb=(1, 1)):
      self.ors = ors
      self.ppc = ppc
      self.cpb = cpb

  def fit(self, x, y = None):
    return self
 
  def transform(self, images):
    ''' Return the HOG of an image as computed in SKIMAGE
    Parameters
    ----------
    images: list of image arrays 

    Returns
    -------
    features: numpy array of hog feature vectors
    '''
    features = []
    for image in images:
      f = hog(image, orientations=self.ors, pixels_per_cell=self.ppc, cells_per_block=self.cpb, visualise=False)
      features.append(f)
    return np.array(features)


class BWTransformer(BaseEstimator, TransformerMixin):
  '''Input list of image arrays and return list of equalized histogram grayscale images

  Parameters and Attributes
  ----------
  None

  Examples
  --------
  from feature_util import BWTransformer
  bw = BWTransformer()
  im = cv2.imread(image_path)
  image_bw = bw.transform(im)
  '''
  def fit(self, x, y = None):
    return self

  def transform(self, images):
    ''' Reutrn equalized grayscale image from image_path
    Parameters
    ----------
    images: list of image arrays

    Returns
    -------
    bw_images: list of images that are grayscaled with equalized histograms
    '''
    bw_images = []
    for image in images:
      imbw = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
      bw_images.append(imbw)

    return bw_images

class DimTransformer(BaseEstimator, TransformerMixin):
  '''Input list of image paths and return image with specified dimensions

  Parameters and Attributes
  ----------
  w: desired image width
  h: desired image height

  Examples
  --------
  from feature_util import DimTransformer
  DT = DimTransformer(w = 100, h = 100)
  resized_images = DT.transform(image_path)
  '''
  def __init__(self, w=100, h=100):
    self.w = w
    self.h = h

  def fit(self, x, y = None):
    return self

  def transform(self, image_paths):
    ''' Read image from image path and return as size self.w by self.h
    Parameters
    ----------
    image_paths: list of path to a valid images

    Returns
    -------
    resized_images: list of image array with width self.w and heigth self.h
    '''
    if type(image_paths) == type('s'): 
      ## if image_paths is a string, assume its a single item list
      image_paths = [image_paths]
    
    resized_images = []
    for image_path in image_paths:
      im = cv2.imread(image_path)
      resized_im = cv2.resize(im, (self.w, self.h))
      resized_images.append(resized_im)
    
    return resized_images


class HSVHistTransformer(BaseEstimator, TransformerMixin):
  '''compute the HSV histogram of a list of images

  Parameters
  ----------
  hist_type: string that is either "hue", "sat", or "val" 
             (corresponding to HSV respectively)

  Returns
  -------
  numpy array of normalized histogram (should sum to 1)

  Examples
  from feature_util import HSVHistTransformer
  HueHist = HSVHistTransformer(hist_type = "hue")
  im = cv.imread(image_path)
  hue_hist = HueHist.transform(im)
  '''
  def __init__(self, hist_type = "hue"):
    self.hist_type = hist_type

  def fit(self, x, y=None):
    return self

  def transform(self, images):
    '''Compute a normalized color histogram of an image
    Parameters
    ----------
    im: an image array

    Returns:
    --------
    normalized_hists: array of normalized color histograms
    '''
    if self.hist_type == "hue":
      num_bins = [180]
      range_bins = [0,180]
      channel = [0]
    if self.hist_type == "sat":
      num_bins = [256]
      range_bins = [0,256]
      channel = [1]
    if self.hist_type == "val":
      num_bins = [256]
      range_bins = [0,256]
      channel = [2]

    normalized_color_hists = []
    for image in images:
      color_hist = cv2.calcHist([image], channel, None, num_bins, range_bins)
      color_hist_normalized = color_hist/(1.0 + sum(color_hist))
      normalized_color_hists.append(np.squeeze(color_hist_normalized))

    return np.array(normalized_color_hists)

