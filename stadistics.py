from math import factorial, sqrt, ceil
from scipy import stats


class Estimator:
    def __init__(self, alpha, n, N):
        self.__alpha = alpha
        self.__n = n
        self.__N = N

    def __ncr(self, r):
        n = self.__n
        f = factorial
        return f(n) // (f(r) * f(n - r))

    def __p_val_bin(self, x, p=0):
        alpha, n, N = self.__alpha, self.__n, self.__N
        pt = x / N
        fmp = 1 - stats.binom.cdf(x, n, pt)
        return fmp < alpha

    def __p_val_norm(self, x):
        normal = stats.norm()
        pg, p0 = x / self.__n, x / self.__N
        p = normal.ppf(1 - self.__alpha)
        z = (pg - p0) / sqrt(p0 * (1 - p0) / self.__n)
        return z > p

    def hyp_test(self, x):
        if x < 5:
            if self.__p_val_bin(x):
                return "P > {}".format(x)
            else:
                return "P = {}".format(x)
        else:
            if self.__p_val_norm(x):
                return "P > {}".format(x)
            else:
                return "P = {}".format(x)

    def conf_int(self, x):
        n = self.__n
        z = stats.norm.ppf(1 - self.__alpha/2)
        p = x / n
        li = p - z * (sqrt(p * (1 - p) / n)) + 1 / (2 * n)
        ls = p + z * (sqrt(p * (1 - p) / n)) + 1 / (2 * n)
        return ceil(self.__N * li), ceil(self.__N * ls)

    @staticmethod
    def fisher(alpha, v1, v2):
        if alpha <= 0:
            return 0

        return round(stats.f.ppf(1 - alpha, v1, v2), 2)
