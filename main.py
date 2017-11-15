import argparse
import bisect
import cv2
import errno
import os
import shutil
import time

import numpy as np
from numba import jit
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from PIL import ImageEnhance, Image


@jit
def im_adjust(src, tol=1, vin=None, v_out=(0, 255)):
    if vin is None:
        vin = [0, 255]
    assert len(src.shape) == 3, 'Input image should be 2-dims'
    tol = max(0, min(100, tol))

    if tol > 0:
        hist = np.histogram(src, bins=list(range(256)), range=(0, 255))[0]
        cum = hist.copy()
        for i in range(1, 256): cum[i] = cum[i - 1] + hist[i]

        total = src.shape[0] * src.shape[1]
        low_bound = total * tol / 100
        upp_bound = total * (100 - tol) / 100
        vin[0] = bisect.bisect_left(cum, low_bound)
        vin[1] = bisect.bisect_left(cum, upp_bound)

    scale = (v_out[1] - v_out[0]) / (vin[1] - vin[0])
    vs = src - vin[0]
    vs[src < vin[0]] = 0
    vd = vs * scale + 0.5 + v_out[0]
    vd[vd > v_out[1]] = v_out[1]
    dst = vd

    return dst


def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


def operate_image(option):
    gray = cv2.cvtColor(option, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    area = Image.fromarray(thresh).convert("RGB").getcolors()[1][0]

    d = ndimage.distance_transform_edt(thresh)
    local_max = peak_local_max(d, indices=False, min_distance=20, labels=thresh)
    markers = ndimage.label(local_max, structure=np.ones((3, 3)))[0]
    labels = watershed(-d, markers, mask=thresh)
    mask = np.zeros(gray.shape, dtype=np.uint8)

    for label in np.unique(labels):
        if label == 0:
            continue
        mask[labels == label] = 255

    c1, c2 = cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    _, contours, _ = cv2.findContours(mask.copy(), c1, c2)
    return contours, mask, area


def apply_watershed(image, crop=False, quantity=0, index=0):
    if crop:
        contours, mask, area = operate_image(image)
        width, height = mask.shape
        total_area = width * height
        img = cv2.imread("enhanced.png")

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            p = w * h * 100 / total_area
            if 0.1 <= p <= 1:
                index, quantity = index + 1, quantity + 1
                ro = img[y: y + h, x: x + w]
                cv2.imwrite("images/test/" + str(index) + '.png', ro)
        return mask, image, quantity, area

    image = im_adjust(image)
    image = image.astype(np.uint8)
    shifted = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    _, mask, area = operate_image(shifted)
    return mask, image, quantity, area


def count(x):
    y = 2.692717401 + 0.001807464 * x
    return int(round(y - 2.616176454, 0)), int(round(y + 2.616176454, 0))


def main():
    if os.path.exists("images/test"):
        shutil.rmtree(os.getcwd() + "/images/test")
    try:
        os.makedirs(os.getcwd() + "/images/test")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    im = Image.open(args["image"])
    contrast = ImageEnhance.Contrast(im)
    contrast.enhance(2)
    image = np.asarray(im, dtype=np.uint8)
    image = adjust_gamma(image)

    mask, image, quantity, _ = apply_watershed(image)
    image = Image.fromarray(image)
    contrast = ImageEnhance.Contrast(image)
    contrast.enhance(3).save("enhanced.png")
    image = cv2.imread("enhanced.png")
    mask, image, quantity, area = apply_watershed(image, crop=True)
    res = count(area)

    print("Total of cells of sample: {}.".format(quantity))
    print("Total of cells of population approximately: {}.".format(res))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image")
    args = vars(ap.parse_args())
    initial = time.time()
    main()
    end = time.time()
    print("Time elapsed: {} seconds.".format(round(end - initial, 2)))
