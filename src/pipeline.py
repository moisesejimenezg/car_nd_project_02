import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np

from calibration import Calibration
from color import Color
from geometry import Point
from geometry import Transformation
from geometry import Trapezoid
from gradient import Gradient
from lines import Lines
from perspective import Perspective


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
        self.calibration_path_ = "../camera_cal/calibration*.jpg"
        self.curvatures_ = {"left": [], "right": []}

    def Calibrate(self, calibration_images_pattern):
        files = glob.glob(calibration_images_pattern)
        for image in files:
            img = plt.imread(image)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            self.calibration_.Update(gray)
        self.calibration_.Calibrate()

    def InitPerspective(self):

        src_p0 = Point(190, 720)
        src_p1 = Point(550, 480)
        src_p2 = Point(730, 480)
        src_p3 = Point(1115, 720)

        src_trp = Trapezoid(src_p0, src_p1, src_p2, src_p3)

        dst_p0 = Point(250, 720)
        dst_p1 = Point(250, 460)
        dst_p2 = Point(980, 460)
        dst_p3 = Point(980, 720)

        dst_trp = Trapezoid(dst_p0, dst_p1, dst_p2, dst_p3)

        transformation = Transformation(src_trp, dst_trp)
        self.perspective_ = Perspective(transformation)

    def Undistort(self, img):
        self.img_ = self.calibration_.Undistort(img)

    def CalculateGradient(self):
        self.gradient_.CalculateGradient(self.img_)

    def FilterGradients(
        self,
        absolute_thresh=(0, 255),
        magnitude_thresh=(0, 255),
        direction_thresh=(0, np.pi / 2),
    ):
        absolute = self.gradient_.AbsoluteThreshold(absolute_thresh)
        self.gradient_images_.append(absolute[0])
        self.gradient_images_.append(absolute[1])
        self.gradient_images_.append(self.gradient_.MagnitudeThreshold(magnitude_thresh))
        self.gradient_images_.append(self.gradient_.DirectionThreshold(direction_thresh))

    def InitColor(self):
        self.color_ = Color(self.img_)

    def FilterColor(self, threshold=(0, 255)):
        self.filtered_ = self.color_.Filter(threshold)

    def JoinOption(self, option="A"):
        combined = np.zeros_like(self.gradient_images_[0])
        abs_threshold = []
        abs_threshold.append(self.gradient_images_[0])
        abs_threshold.append(self.gradient_images_[1])
        mag_threshold = self.gradient_images_[2]
        dir_threshold = self.gradient_images_[3]
        if option is "A":
            combined[
                (((abs_threshold[0] == 1) & (abs_threshold[1] == 1)) | ((mag_threshold == 1) & (dir_threshold == 1)))
                | (self.filtered_ == 1)
            ] = 1
        elif option is "B":
            combined[(abs_threshold[0] == 1) | (self.filtered_ == 1)] = 1
        elif option is "C":
            combined[(abs_threshold[0] == 1) | (self.filtered_ == 1)] = 1
        return combined

    def Transform(self, img):
        return self.perspective_.Transform(img)

    def FitPolynomial(self, img, visualize=False):
        return self.lines_.Process(img, visualize)

    def CalculateCurvature(self, polynomial_fit, y, ym_per_pix=(30 / 720)):
        return self.lines_.CalculateCurvature(polynomial_fit, y, ym_per_pix)

    def PlotLaneOnImage(self, img, left_fit, right_fit):
        lane_img = self.lines_.PlotPoly(img, left_fit, right_fit)
        unwarped_lane = self.perspective_.InverseTransform(lane_img)
        return cv2.addWeighted(self.img_, 1, unwarped_lane, 0.3, 0)

    def Prepare(self):
        self.Calibrate(self.calibration_path_)
        self.InitPerspective()

    def Process(self, img, display=False):
        self.gradient_images_ = []
        if display:
            plt.imshow(img)
            plt.show()
        self.Undistort(img)
        self.CalculateGradient()
        self.FilterGradients((20, 100), (30, 100), (0.7, 1.3))
        self.InitColor()
        self.FilterColor((170, 255))
        combinedA = self.JoinOption()

        transformed = self.Transform(combinedA)

        left_fit, right_fit = self.FitPolynomial(transformed, False)

        result = self.PlotLaneOnImage(self.img_, left_fit, right_fit)

        y_eval = img.shape[0]
        self.curvatures_["left"].append(self.CalculateCurvature(left_fit.rw_polynomial_, y_eval))
        self.curvatures_["right"].append(self.CalculateCurvature(right_fit.rw_polynomial_, y_eval))

        return result

    def GetCurvatures(self):
        return self.curvatures_
