import mahotas as mh
import numpy as np


class CellCounter:
    def __init__(self, image):
        self.__image = image

    def count(self):
        dna = self.__image
        dna = dna.max(axis=2)

        dnaf = mh.gaussian_filter(dna, 2.)
        T_mean = dnaf.mean()
        bin_image = dnaf > T_mean

        labeled, nr_objects = mh.label(bin_image)

        sigma = 12.0
        dnaf = mh.gaussian_filter(dna.astype(float), sigma)
        maxima = mh.regmax(mh.stretch(dnaf))
        maxima, _ = mh.label(maxima)

        dist = mh.distance(bin_image)
        dist = 255 - mh.stretch(dist)
        watershed = mh.cwatershed(dist, maxima)
        watershed *= bin_image
        watershed = mh.labeled.remove_bordering(watershed)

        sizes = mh.labeled.labeled_size(watershed)
        watershed = watershed.astype(np.intc)

        min_size = 1000
        filtered = mh.labeled.remove_regions_where(watershed, sizes < min_size)
        return nr_objects
