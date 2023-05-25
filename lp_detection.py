import sys, os
import keras
import cv2
import traceback

from src.keras_utils 			import load_model
from glob 						import glob
from os.path 					import splitext, basename
from src.utils 					import im2single
from src.keras_utils 			import load_model, detect_lp
from src.label 					import Shape, writeShapes


def adjust_pts(pts,lroi):
	return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))


class LicensePlateDetector:
  def __init__(self, lp_threshold=.5, wpod_net_path = "weights/lp-detector/wpod-net_update1.h5"):
    self.lp_threshold = lp_threshold

    self.wpod_net = None
    self.load_model_wpod(wpod_net_path)
    
  def load_model_wpod(self, wpod_net_path):
    try:
      self.wpod_net = load_model(wpod_net_path)
    except:
      print('Can not load wpod net')

  def detect(self, image):
    try:
      # print('Searching for license plates using WPOD-NET')
      # print('\t Processing')

      Ivehicle = image

      ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
      side  = int(ratio*288.)
      bound_dim = min(side + (side%(2**4)),608)
      # print("\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio))

      Llp,LlpImgs,_ = detect_lp(self.wpod_net,im2single(Ivehicle),bound_dim,2**4,(120,100),self.lp_threshold)

      if len(LlpImgs):
        Ilp = LlpImgs[0]
        # Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
        # Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

        s = Shape(Llp[0].pts)

        # cv2.imwrite('%s_lp.png' % (bname),Ilp*255.)
        # writeShapes('%s_lp.txt' % (bname),[s])

        return Ilp*255., [s]

    except:
      traceback.print_exc()
    
    return None, None



if __name__ == '__main__':

  detector = LicensePlateDetector(.5, "weights/lp-detector/wpod-net_update1.h5")

  img = cv2.imread('/Users/lasion/Library/CloudStorage/GoogleDrive-lxytb07@gmail.com/My Drive/NCKH/NCKH_2023/data/GreenParking/0000_06886_b.jpg')
  lp_image, lp_labels = detector.detect(img)

  if lp_image is not None:
    cv2.imwrite('lp_img.jpg', lp_image)
    writeShapes('lp_lb.txt', lp_labels)
    print('Saved result')

  sys.exit(0)


