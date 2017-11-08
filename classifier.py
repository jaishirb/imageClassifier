import cv2
import itertools
from math import hypot


class Classifier:
    def __init__(self, image):
        self.__im = image
        self.__height, self.__width, _ = self.__im.shape
        self.__x, self.__y = 0, 0

    def __calc_center(self):
        x = self.__width / 2 if self.__width % 2 == 0 else (self.__width + 1) / 2
        y = self.__height / 2 if self.__height % 2 == 0 else (self.__height + 1) / 2
        self.__x, self.__y = x, y

    def dist(self, p):
        return hypot(p[1] - p[0], self.__y - self.__x)

    def classify(self, id):
        gray = cv2.cvtColor(self.__im, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
        width, height = thresh.shape
        self.__calc_center()

        upper, lower = max([width, height]), min([width, height])
        index = 0 if lower == width else 1
        nums = [i for i in range(upper)]
        data, dists = [], []

        for value in itertools.product(nums, repeat=2):
            if value[index] < lower and thresh[value[0], value[1]] == 0:
                dists.append(self.dist(value))

