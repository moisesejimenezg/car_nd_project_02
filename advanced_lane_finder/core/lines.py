import cv2
import matplotlib.pyplot as plt
import numpy as np


class Window:
    def __init__(self, x_low, x_high, y_low, y_high):
        self.x_low_ = x_low
        self.x_high_ = x_high
        self.y_low_ = y_low
        self.y_high_ = y_high


class PolynomialFit:
    def __init__(self, polynomial, rw_polynomial, polynomial_fit, ploty):
        self.polynomial_ = polynomial
        self.rw_polynomial_ = rw_polynomial
        self.polynomial_fit_ = polynomial_fit
        self.ploty_ = ploty


class Lines:
    def __init__(self, windows=9, margin=100, minimum_hits=50):
        self.windows_ = windows
        self.margin_ = margin
        self.minimum_hits_ = minimum_hits

    def __DrawWindow__(self, out_img, window):
        cv2.rectangle(
            out_img,
            (window.x_low_, window.y_low_),
            (window.x_high_, window.y_high_),
            (0, 255, 0),
            2,
        )

    def __FindValidIndices__(self, nonzerox, nonzeroy, window):
        return (
            (nonzeroy >= window.y_low_)
            & (nonzeroy < window.y_high_)
            & (nonzerox >= window.x_low_)
            & (nonzerox < window.x_high_)
        ).nonzero()[0]

    def __ProcessWindows__(
        self,
        img,
        leftx_base,
        rightx_base,
        window,
        window_height,
        nonzerox,
        nonzeroy,
        left_lane_ids,
        right_lane_ids,
        out_img,
    ):
        leftx_current = leftx_base
        rightx_current = rightx_base
        y_low = img.shape[0] - (window + 1) * window_height
        y_high = img.shape[0] - window * window_height
        left_window = Window(leftx_current - self.margin_, leftx_current + self.margin_, y_low, y_high)
        right_window = Window(rightx_current - self.margin_, rightx_current + self.margin_, y_low, y_high)
        self.__DrawWindow__(out_img, left_window)
        self.__DrawWindow__(out_img, right_window)
        good_left_ids = self.__FindValidIndices__(nonzerox, nonzeroy, left_window)
        good_right_ids = self.__FindValidIndices__(nonzerox, nonzeroy, right_window)
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
            print("Could not fit polynomial")
            fit = 1 * ploty ** 2 + 1 * ploty
        return PolynomialFit(polynomial, rw_polynomial, fit, ploty)

    def __Visualize__(self, out_img, lefty, leftx, righty, rightx, left_fit, right_fit, ploty):
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        plt.plot(left_fit, ploty, color="yellow")
        plt.plot(right_fit, ploty, color="yellow")
        plt.imshow(out_img)
        plt.show()

    def PlotPoly(self, img, left_fit, right_fit):
        pts_left = np.array([np.transpose(np.vstack([left_fit.polynomial_fit_, left_fit.ploty_]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit.polynomial_fit_, right_fit.ploty_])))])
        pts = np.hstack((pts_left, pts_right))
        out_img = np.zeros_like(img).astype(np.uint8)
        cv2.fillPoly(out_img, np.int_([pts]), (0, 255, 0))
        return out_img

    def Process(self, img, visualize=False, xm_per_pix=(3.7 / 900), ym_per_pix=(30 / 720)):
        histogram = np.sum(img[img.shape[0] // 2 :, :], axis=0)
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        left_lane_ids = []
        right_lane_ids = []
        out_img = np.dstack((img, img, img))
        window_height = np.int(img.shape[0] // self.windows_)
        leftx_current = leftx_base
        rightx_current = rightx_base
        for window in range(self.windows_):
            leftx_current, rightx_current = self.__ProcessWindows__(
                img,
                leftx_current,
                rightx_current,
                window,
                window_height,
                nonzerox,
                nonzeroy,
                left_lane_ids,
                right_lane_ids,
                out_img,
            )
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
        left_fit = self.__FitPolynomial__(leftx, lefty, ploty, xm_per_pix, ym_per_pix)
        right_fit = self.__FitPolynomial__(rightx, righty, ploty, xm_per_pix, ym_per_pix)
        if visualize:
            self.__Visualize__(
                out_img,
                lefty,
                leftx,
                righty,
                rightx,
                left_fit.polynomial_fit_,
                right_fit.polynomial_fit_,
                ploty,
            )
        return left_fit, right_fit

    def LookBack(self, binary_warped, left_fit, right_fit, margin=100, xm_per_pix=(3.7 / 900), ym_per_pix=(30 / 720)):
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = (nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
            nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)
        )
        right_lane_inds = (
            nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)
        ) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin))

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fit = self.__FitPolynomial__(leftx, lefty, ploty, xm_per_pix, ym_per_pix)
        right_fit = self.__FitPolynomial__(rightx, righty, ploty, xm_per_pix, ym_per_pix)
        return left_fit, right_fit

    def CalculateCurvature(self, polynomial_fit, y, ym_per_pix=(30 / 720)):
        numerator = (1 + (2 * polynomial_fit[0] * y * ym_per_pix + polynomial_fit[1]) ** 2) ** (3 / 2)
        denominator = abs(2 * polynomial_fit[0])
        return numerator / denominator

    def CalculateOffsetFromCenter(self, left_fit, right_fit, xm_per_pix=(3.7 / 900)):
        left_x = left_fit.polynomial_[0] * 720 ** 2 + left_fit.polynomial_[1] * 720 + left_fit.polynomial_[2]
        right_x = right_fit.polynomial_[0] * 720 ** 2 + right_fit.polynomial_[1] * 720 + right_fit.polynomial_[2]
        offset = (left_x + right_x) / 2 - 640
        return offset * xm_per_pix
