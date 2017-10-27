import mahotas as mh


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

        return nr_objects
