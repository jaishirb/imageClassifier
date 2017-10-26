from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage, stats
from numba import jit
import numpy as np
import argparse
import cv2
import bisect
import math
import time
import os
import errno
import shutil
from statistics import mode
from PIL import Image
from cellCounter import cellCounter


@jit
def imadjust(src, tol=1, vin=[0, 255], vout=(0, 255)):
    assert len(src.shape) == 3 ,'Input image should be 2-dims'
    tol = max(0, min(100, tol))

    if tol > 0:
        hist = np.histogram(src,bins =list(range(256)),range=(0,255))[0]
        cum = hist.copy()
        for i in range(1, 256): cum[i] = cum[i - 1] + hist[i]

        total = src.shape[0] * src.shape[1]
        low_bound = total * tol / 100
        upp_bound = total * (100 - tol) / 100
        vin[0] = bisect.bisect_left(cum, low_bound)
        vin[1] = bisect.bisect_left(cum, upp_bound)

    scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
    vs = src-vin[0]
    vs[src<vin[0]]=0
    vd = vs*scale+0.5 + vout[0]
    vd[vd>vout[1]] = vout[1]
    dst = vd

    return dst


def dist(data):
    x, y, z = data
    return math.sqrt(math.pow(x-255,2) + math.pow(y-255,2) + math.pow(z-255,2))


def brute_force(img):
    width, height, channels = img.shape	
    img = img[:,:,0]
    img[img >= 190] = 0
    img[img > 0] = 255
    # cv2.imwrite("images/img.jpg", img)
    
    """
    cv2.imshow("img",img)
    k = cv2.waitkey(0)
    if k == 27:
        cv2.destroyAllWindows()	
    """
    """
    cv2.imshow("gray", gray)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
        
    for y in range(0, height):
        for x in range(0, width):
            RGB = img[x,y]
            if dist(list(RGB)) < 40:
                img[x,y] = np.array([0, 0, 0])
            else:
                img[x,y] = np.array([255, 255, 255])
    cv2.imshow("img", img)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
    """

def operate_image(option):

    gray = cv2.cvtColor(option, cv2.COLOR_BGR2GRAY)	
    thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    kernel = np.ones((2,2),np.uint8)
    thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2)

    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=20,labels=thresh)
     
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)

    mask = np.zeros(gray.shape, dtype="uint8")
    index = 0
    
    for label in np.unique(labels):
        if label == 0:
            continue
        mask[labels == label] = 255
        

    im2, cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE)

    return im2, cnts, hierarchy, mask


def apply_watershed(image, crop=False):

    if not crop:
        image = imadjust(image)
        cv2.imwrite("images/npic.jpg",image)
        image = cv2.imread("images/npic.jpg")
        shifted = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)		
        im2, cnts, hierarchy, mask = operate_image(shifted)
        gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
        quantity = 0
    
    else:
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        im2, cnts, hierarchy, mask = operate_image(image)
        quantity, index = 0, 0

        mask_size = lambda mask: tuple(mask.shape[1::-1])
        temp_area = mask_size(mask)
        total_area = temp_area[0]*temp_area[1]
        # s_dataset, p_dataset = [], []
        timg = cv2.imread(args["image"])

        for cnt in cnts:
            x,y,w,h = cv2.boundingRect(cnt)
            p = w*h*100/total_area
            if p >= 0.2 and p <= 1:
                # s_dataset.append(p)
                index, quantity = index+1, quantity+1
                roi=timg[y: y + h, x: x + w]
                cv2.imwrite("images/test/" + str(index) + '.png', roi)
            # p_dataset.append(p)

    return mask, image, quantity


def main():
    try:
        if os.path.exists("images/test"):
            shutil.rmtree(os.getcwd() + "/images/test")
        try:
            os.makedirs(os.getcwd() + "/images/test")
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        image = cv2.imread(args["image"])
        mask, image, quantity = apply_watershed(image)

        _,alpha = cv2.threshold(mask,0,255,cv2.THRESH_BINARY_INV)    
        b, g, r = cv2.split(image)
        rgba = [b,g,r, alpha]
        dst = cv2.merge(rgba,4)

        cv2.imwrite("images/test_transparent.png", dst)
        image = cv2.imread("images/test_transparent.png")
        mask, image, quantity = apply_watershed(image)

        if os.path.exists("images/test_transparent.png"):
            os.remove(os.getcwd() + "/images/test_transparent.png")

        _,alpha = cv2.threshold(mask,0,255,cv2.THRESH_BINARY_INV)    
        b, g, r = cv2.split(image)
        rgba = [b,g,r, alpha]
        dst = cv2.merge(rgba,4)

        image = cv2.cvtColor(dst, cv2.COLOR_RGBA2RGB)
        brute_force(image)

        image = np.asarray(image)
        mask, image, quantity = apply_watershed(image, crop=True)
        cell_counter = cellCounter(image)

        print("Total of cells of sample: {}.".format(quantity))
        print("Total of cells of population aproximately: {}".format(cell_counter.count()))
    except:
        main()


if __name__ == '__main__':
    initial = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,help="path to input image")
    args = vars(ap.parse_args())
    main()
    end = time.time()
    print("Time elapsed: {}".format(end - initial))
        
