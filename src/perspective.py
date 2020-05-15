import cv2

class Perspective:
    def __init__(self, transformation):
        self.transformation_ = transformation
        self.m_ = cv2.getPerspectiveTransform(self.transformation_.GetSource(), self.transformation_.GetDestination())
        self.m_inv_ = cv2.getPerspectiveTransform(self.transformation_.GetDestination(), self.transformation_.GetSource())

    def transform(self, img):
        img_size = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(img, self.m_, img_size, flags=cv2.INTER_LINEAR)

    def inverse_transform(self, img):
        img_size = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(img, self.m_inv_, img_size, flags=cv2.INTER_LINEAR)
