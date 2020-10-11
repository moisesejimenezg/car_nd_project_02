import cv2
import numpy as np


class Gradient:
    def __init__(self, kernel_size=3):
        self.kernel_size_ = kernel_size
        self.x_ = []
        self.y_ = []

    def CalculateGradient(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        self.x_ = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.kernel_size_)
        self.y_ = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.kernel_size_)

    def Scale(self, img, peak=255):
        return np.uint8(peak * img / np.max(img))

    def AbsoluteThreshold(self, thresh=(0, 255)):
        absolute = (np.absolute(self.x_), np.absolute(self.y_))
        scaled_abs = (self.Scale(absolute[0]), self.Scale(absolute[1]))
        output = np.zeros_like(scaled_abs)
        output[0, (scaled_abs[0] > thresh[0]) & (scaled_abs[0] < thresh[1])] = 1
        output[1, (scaled_abs[1] > thresh[0]) & (scaled_abs[1] < thresh[1])] = 1
        return output

    def MagnitudeThreshold(self, thresh=(0, 255)):
        magnitude = np.sqrt(self.x_ ** 2 + self.y_ ** 2)
        scaled_mag = self.Scale(magnitude)
        output = np.zeros_like(scaled_mag)
        output[(scaled_mag > thresh[0]) & (scaled_mag < thresh[1])] = 1
        return output

    def DirectionThreshold(self, thresh=(0, np.pi / 2)):
        absolute = (np.absolute(self.x_), np.absolute(self.y_))
        atan_abs_sob_gra = np.arctan2(absolute[1], absolute[0])
        output = np.zeros_like(self.x_)
        output[(atan_abs_sob_gra > thresh[0]) & (atan_abs_sob_gra < thresh[1])] = 1
        return output
