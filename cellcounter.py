from PIL import Image
from numpy import sqrt

import cv2
from math import pi


class CellCounter:
    def __init__(self, image):
        self.__image = image

    def count(self, data, quantity):

        img = cv2.imread('enh.png')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, 1)
        _, contours, h = cv2.findContours(thresh, 1, 2)

        for cnt in contours:
            cv2.drawContours(img, [cnt], 0, (0, 255, 255), 1)

        radius = 0
        for i in data:
            img2 = Image.fromarray(i).convert("RGB")
            colors = img2.getcolors()
            radius += sqrt(colors[1][0] / pi)

        radius = round(radius / len(data)) - 0.2

        h, w, c = img.shape
        px, cont = Image.fromarray(img).load(), 0
        for y in range(h):
            for x in range(w):
                if px[x, y] == (0, 255, 255):
                    cont += 1

        est = round(cont / (2.5 * pi * radius))
        est = est if est > quantity else round(cont / (2.5 * pi * (radius - 1)))

        return int(est)
