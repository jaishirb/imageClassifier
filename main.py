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
from cellcounter import CellCounter
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


def brute_force(img):
    img = img[:, :, 0]
    img[img >= 185] = 0
    img[img > 0] = 255


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
    mask = np.zeros(gray.shape, dtype="uint8")

    for label in np.unique(labels):
        if label == 0:
            continue
        mask[labels == label] = 255

    c1, c2 = cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    im2, contours, hierarchy = cv2.findContours(mask.copy(), c1, c2)
    return im2, contours, hierarchy, mask, area


def apply_watershed(image, crop=False):
    data, t = [], 0
    if not crop:
        image = im_adjust(image)
        cv2.imwrite("images/pic.jpg", image)
        image = cv2.imread("images/pic.jpg")
        shifted = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        im2, contours, hierarchy, mask, area = operate_image(shifted)
        quantity = 0
    else:
        im2, contours, hierarchy, mask, area = operate_image(image)
        quantity, index = 0, 0
        width, height = mask.shape
        total_area = width * height
        img = cv2.imread("enh.png")

        local_area = 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            p = w * h * 100 / total_area
            if 0.1 <= p <= 1:
                index, quantity = index + 1, quantity + 1
                roi = mask[y: y + h, x: x + w]
                local_area += Image.fromarray(roi).convert("RGB").getcolors()[1][0]
                ro = img[y: y + h, x: x + w]
                data.append(roi)
                cv2.imwrite("images/test/" + str(index) + '.png', ro)
        local_area /= len(data)
        t = round(area/local_area)
    return mask, image, quantity, data, t


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
    contrast.enhance(3).save("enh.png")
    image = cv2.imread("enh.png")
    image = adjust_gamma(image)

    mask, image, quantity, _, _ = apply_watershed(image)
    _, alpha = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV)
    b, g, r = cv2.split(image)
    rgba = [b, g, r, alpha]
    dst = cv2.merge(rgba, 4)

    contrast = ImageEnhance.Contrast(Image.fromarray(dst))
    contrast.enhance(3).save("enhanced.png")
    dst = cv2.imread("enhanced.png")

    cv2.imwrite("images/test_transparent.png", dst)
    image = cv2.imread("images/test_transparent.png")
    mask, image, quantity, _, _ = apply_watershed(image)

    if os.path.exists("images/test_transparent.png"):
        os.remove(os.getcwd() + "/images/test_transparent.png")

    if os.path.exists("images/pic.jpg"):
        os.remove(os.getcwd() + "/images/pic.jpg")

    _, alpha = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV)
    b, g, r = cv2.split(image)
    rgba = [b, g, r, alpha]
    dst = cv2.merge(rgba, 4)

    image = cv2.cvtColor(dst, cv2.COLOR_RGBA2RGB)

    image = np.asarray(image)
    mask, image, quantity, data, t = apply_watershed(image, crop=True)
    cell_counter = CellCounter(image)
    res = cell_counter.count(data, quantity)
    res = int((res + quantity + t)*1.025/3)

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
