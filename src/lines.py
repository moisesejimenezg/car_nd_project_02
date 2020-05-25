import cv2
import matplotlib.pyplot as plt
import numpy as np


class Window:
    def __init__(self, x_low, x_high, y_low, y_high):
        self.x_low_ = x_low
        self.x_high_ = x_high
        self.y_low_ = y_low
        self.y_high_ = y_high


class Lines:
    def __init__(self, windows=9, margin=100, minimum_hits=50):
        self.windows_ = windows
        self.margin_ = margin
        self.minimum_hits_ = minimum_hits

    def __DrawWindow__(self, out_img, window):
        cv2.rectangle(out_img, (window.x_low_, window.y_low_),
                      (window.x_high_, window.y_high_), (0, 255, 0), 2)

    def __FindValidIndices__(self, nonzerox, nonzeroy, window):
        return ((nonzeroy >= window.y_low_) & (nonzeroy < window.y_high_) & (nonzerox >= window.x_low_) & (nonzerox < window.x_high_)).nonzero()[0]

    def __ProcessWindows__(self, img, leftx_base, rightx_base, window, window_height, nonzerox, nonzeroy, left_lane_ids, right_lane_ids, out_img):
        leftx_current = leftx_base
        rightx_current = rightx_base
        y_low = img.shape[0] - (window + 1) * window_height
        y_high = img.shape[0] - window * window_height
        left_window = Window(leftx_current - self.margin_,
                             leftx_current + self.margin_, y_low, y_high)
        right_window = Window(rightx_current - self.margin_,
                              rightx_current + self.margin_, y_low, y_high)
        self.__DrawWindow__(out_img, left_window)
        self.__DrawWindow__(out_img, right_window)
        good_left_ids = self.__FindValidIndices__(
            nonzerox, nonzeroy, left_window)
        good_right_ids = self.__FindValidIndices__(
            nonzerox, nonzeroy, right_window)
        left_lane_ids.append(good_left_ids)
        right_lane_ids.append(good_right_ids)
        if len(good_left_ids) > self.minimum_hits_:
            leftx_current = np.int(np.mean(nonzerox[good_left_ids]))
        if len(good_right_ids) > self.minimum_hits_:
            rightx_current = np.int(np.mean(nonzerox[good_right_ids]))
        return leftx_current, rightx_current

    def __FitPolynomial__(self, x, y, ploty, xm_per_pix, ym_per_pix):
        rw_polynomial = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)
        polynomial = np.polyfit(y, x, 2)
        try:
            fit = polynomial[0] * ploty ** 2 + polynomial[1] * ploty + polynomial[2]
        except TypeError:
            print('Could not fit polynomial')
            fit = 1 * ploty ** 2 + 1 * ploty
        return polynomial, rw_polynomial, fit

    def __Visualize__(self, out_img, lefty, leftx, righty, rightx, left_fit, right_fit, ploty):
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]
        plt.plot(left_fit, ploty, color='yellow')
        plt.plot(right_fit, ploty, color='yellow')
        plt.imshow(out_img)
        plt.show()

    def Process(self, img, visualize=False, xm_per_pix = (3.7/900), ym_per_pix = (30/720)):
        histogram = np.sum(img[img.shape[0]//2:, :], axis=0)
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        left_lane_ids = []
        right_lane_ids = []
        out_img = np.dstack((img, img, img))
        window_height = np.int(img.shape[0]//self.windows_)
        leftx_current = leftx_base
        rightx_current = rightx_base
        for window in range(self.windows_):
            leftx_current, rightx_current = self.__ProcessWindows__(img, leftx_current, rightx_current, window, window_height,
                                                                    nonzerox, nonzeroy, left_lane_ids, right_lane_ids, out_img)
        try:
            left_lane_ids = np.concatenate(left_lane_ids)
            right_lane_ids = np.concatenate(right_lane_ids)
        except ValueError:
            pass

        leftx = nonzerox[left_lane_ids]
        lefty = nonzeroy[left_lane_ids]
        rightx = nonzerox[right_lane_ids]
        righty = nonzeroy[right_lane_ids]

        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_polynomial, rw_left_polynomial, left_fit = self.__FitPolynomial__(leftx, lefty, ploty, xm_per_pix, ym_per_pix)
        right_polynomial, rw_right_polynomial, right_fit = self.__FitPolynomial__(rightx, righty, ploty, xm_per_pix, ym_per_pix)
        if visualize:
            self.__Visualize__(out_img, lefty, leftx, righty,
                               rightx, left_fit, right_fit, ploty)
        return left_polynomial, rw_left_polynomial, left_fit, right_polynomial, rw_right_polynomial, right_fit

    def CalculateCurvature(self, polynomial_fit, y, ym_per_pix = (30/720)):
        numerator = (1 + (2 * polynomial_fit[0] * y * ym_per_pix + polynomial_fit[1]) ** 2) ** (3/2)
        denominator = abs(2 * polynomial_fit[0])
        return numerator / denominator
