import cv2
import glob
import matplotlib.pyplot as plt

from calibration import Calibration

class Pipeline:
    def __init__(self, nx, ny):
        self.calibration_ = Calibration(nx, ny)
        self.images_ = []

    def Calibrate(self, calibration_images_pattern):
        files = glob.glob(calibration_images_pattern)
        for image in files:
            img = plt.imread(image)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            self.calibration_.Update(gray)
        self.calibration_.Calibrate()

    def Undistort(self, files_pattern):
        files = glob.glob(files_pattern)
        for image in files:
            img = plt.imread(image)
            self.images_.append(self.calibration_.Undistort(img))

