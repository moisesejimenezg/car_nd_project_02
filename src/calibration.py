import cv2
import numpy as np


class Calibration:
    def __init__(self, nx, ny):
        self.nx_ = nx
        self.ny_ = ny
        self.image_points_ = []
        self.object_points_ = []
        self.matrix = []
        self.distortion_coefficients_ = []
        self.shape_ = []

    def Update(self, image):
        ret, corners = cv2.findChessboardCorners(image, (self.nx_, self.ny_), None)
        if ret == True:
            if len(self.shape_) == 0:
                self.shape_ = image.shape[::-1]
            object_points = np.zeros((self.nx_ * self.ny_, 3), np.float32)
            object_points[:, :2] = np.mgrid[0 : self.nx_, 0 : self.ny_].T.reshape(-1, 2)
            self.image_points_.append(corners)
            self.object_points_.append(object_points)

    def Calibrate(self):
        ret, matrix, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.object_points_, self.image_points_, self.shape_, None, None
        )
        self.matrix_ = matrix
        self.distortion_coefficients_ = dist

    def Undistort(self, image):
        return cv2.undistort(image, self.matrix_, self.distortion_coefficients_, None, self.matrix_)
