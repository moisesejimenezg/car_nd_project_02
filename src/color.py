import cv2
import numpy as np

class Color:
    def __init__(self, img):
        self.s_ = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:,:,2]

    def filter(self, threshold=(0, 255)):
        output = np.zeros_like(self.s_)
        output[(self.s_ > threshold[0]) & (self.s_ < threshold[1])] = 1
        return output
