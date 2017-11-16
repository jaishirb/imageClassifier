from math import factorial, sqrt, ceil, sin, asin
from scipy import stats


class Estimator:
    def __init__(self, alpha, n, N):
        self.__alpha = alpha
        self.__n = n
        self.__N = N

    def normal_est(self, x):
        normal = stats.norm()
        z = normal.ppf(1-self.__alpha/2)
        p = x/self.__n
        a = 1/(1 + 1/(self.__n*z**2))
        b = p + (1/(2*self.__n))*z**2
        li = a*(b - sqrt((1/self.__n)*p*(1-p) + (1/(4*self.__n**2))*z**2))
        ls = a*(b + sqrt((1/self.__n)*p*(1-p) + (1/(4*self.__n**2))*z**2))
        return int(li*self.__N), int(ls*self.__N)

    def arcsin_est(self, x):
        normal = stats.norm()
        z = normal.ppf(1 - self.__alpha / 2)
        p = x/self.__n
        li = sin(asin(sqrt(p)) - z/(2*sqrt(self.__n)))**2
        ls = sin(asin(sqrt(p)) + z/(2*sqrt(self.__n)))**2
        return int(li*self.__N, 0), int(ls*self.__N, 0)