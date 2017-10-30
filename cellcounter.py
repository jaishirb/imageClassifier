import mahotas as mh


class CellCounter:
    def __init__(self, image):
        self.__image = image

    def count(self):
        dna = self.__image
        dna = dna.max(axis=2)

        d_naf = mh.gaussian_filter(dna, 2.)
        t_mean = d_naf.mean()

        bin_image = d_naf > t_mean
        labeled, nr_objects = mh.label(bin_image)
        return nr_objects
