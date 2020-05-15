import numpy as np
import cv2

class Point:
    def __init__(self, x, y):
        self.x_ = x
        self.y_ = y

    def GetX(self):
        return self.x_

    def GetY(self):
        return self.y_

class Trapezoid:
    def __init__(self, p0, p1, p2, p3):
        self.p0_ = p0
        self.p1_ = p1
        self.p2_ = p2
        self.p3_ = p3

class Transformation:
    def __init__(self, src, dst):
        self.src_ = src
        self.dst_ = dst

    def GetSource(self):
        result = np.float32([[self.src_.p0_.x_, self.src_.p0_.y_], [self.src_.p1_.x_, self.src_.p1_.y_], [self.src_.p2_.x_, self.src_.p2_.y_], [self.src_.p3_.x_, self.src_.p3_.y_]])
        return result

    def GetDestination(self):
        result = np.float32([[self.dst_.p0_.x_, self.dst_.p0_.y_], [self.dst_.p1_.x_, self.dst_.p1_.y_], [self.dst_.p2_.x_, self.dst_.p2_.y_], [self.dst_.p3_.x_, self.dst_.p3_.y_]])
        return result

