import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np

from calibration import Calibration
from color import Color
from gradient import Gradient
from perspective import Perspective
from lines import Lines


class Pipeline:
    def __init__(self, nx, ny, kernel_size=3):
        self.img_ = {}
        self.calibration_ = Calibration(nx, ny)
        self.gradient_ = Gradient(kernel_size)
        self.lines_ = Lines()
        self.color_ = {}
        self.perspective_ = {}
        self.gradient_images_ = []
        self.filtered_ = {}

    def Calibrate(self, calibration_images_pattern):
        files = glob.glob(calibration_images_pattern)
        for image in files:
            img = plt.imread(image)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            self.calibration_.Update(gray)
        self.calibration_.Calibrate()

    def Undistort(self, img):
        self.img_ = self.calibration_.Undistort(img)

    def CalculateGradient(self):
        self.gradient_.CalculateGradient(self.img_)

    def FilterGradients(self, absolute_thresh=(0, 255), magnitude_thresh=(0, 255), direction_thresh=(0, np.pi/2)):
        absolute = self.gradient_.AbsoluteThreshold(absolute_thresh)
        self.gradient_images_.append(absolute[0])
        self.gradient_images_.append(absolute[1])
        self.gradient_images_.append(
            self.gradient_.MagnitudeThreshold(magnitude_thresh))
        self.gradient_images_.append(
            self.gradient_.DirectionThreshold(direction_thresh))

    def InitColor(self):
        self.color_ = Color(self.img_)

    def FilterColor(self, threshold=(0, 255)):
        self.filtered_ = self.color_.Filter(threshold)

    def JoinOption(self, option='A'):
        combined = np.zeros_like(self.gradient_images_[0])
        abs_threshold = []
        abs_threshold.append(self.gradient_images_[0])
        abs_threshold.append(self.gradient_images_[1])
        mag_threshold = self.gradient_images_[2]
        dir_threshold = self.gradient_images_[3]
        if option is 'A':
            combined[(((abs_threshold[0] == 1) & (abs_threshold[1] == 1)) | (
                (mag_threshold == 1) & (dir_threshold == 1))) | (self.filtered_ == 1)] = 1
        elif option is 'B':
            combined[(abs_threshold[0] == 1) | (self.filtered_ == 1)] = 1
        elif option is 'C':
            combined[(abs_threshold[0] == 1) | (self.filtered_ == 1)] = 1
        return combined

    def InitPerspective(self, transformation):
        self.perspective_ = Perspective(transformation)

    def Transform(self, img):
        return self.perspective_.Transform(img)

    def FitPolynomial(self, img, visualize=False):
        return self.lines_.Process(img, visualize)
