import numpy as np
from operator import itemgetter
import cv2


class Clasificador:

    def __init__(self, imagen, nombre, distancias, colores):

        """ Constructor of Classificador.

            Parameters

            imagen -- image to analize
            nombre -- name of image 
            distancias -- list of distances between nucleus's point and coordinate 
            of minimum color
            colores -- list of coordinates of nucleus's point

        """

        self.__img = imagen
        self.__name = nombre
        self.__dist_lista = distancias
        self.__col_lista = colores

    def guardar_imagen(self, tipo):

        """ Save a image.

            Parameters

            tipo -- phase of cell 

        """


        k = tipo+'/imagen' + str(self.__name) + '.png'
        cv2.imwrite(k, self.__img)

    def dist(self,x, y):

        """ Find euclidean distance between tuples.

            return 
            euclidean distance

            Parameters
            x -- first tuple
            y -- second tuple  

        """

        x1, y1 = x
        x2, y2 = y
        return np.sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2))

    def media(self):

        """ Find mean of distance's list.

            return 
            mean of distance's list 

        """

        return np.mean(self.__dist_lista)

    def varianza(self):

        """ Find variance of distance's list.

            return 
            variance of distance's list 

        """

        return np.std(self.__dist_lista)

    def telofase(self):

        """ Find out if cell is in telophase or not.

            return 
            True if cell is in telophase
            False if cell is not in telophase

        """


        var = self.varianza()
        if var >= 6.5:
            self.guardar_imagen('telofase')
            return True
        return False

    def find_min(self):

        """ Find min of list of distances.

            return 
            min of x and min of y 

        """


        min_x = 1000
        min_y = 1000
        k = len(self.__col_lista)
        for i in range(k):
            x, y = self.__col_lista[i]
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
        return min_x, min_y

    def find_max(self):

        """ Find max list of distances.

            return 
            max of x and max of y 

        """

        max_x = -10
        max_y = -10
        k = len(self.__col_lista)
        for i in range(k):
            x, y = self.__col_lista[i]
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y
        return max_x, max_y

    def find_centroid_cell(self):


        """ Find centroid of nucleus cell.

            return 
            coordinate of nucleus cell's centroid 

        """

        x_min, y_min = self.find_min()
        x_max, y_max = self.find_max()
        x_centroid = int((x_max+x_min)/2)
        y_centroid = int((y_max+y_min)/2)
        centroide = x_centroid, y_centroid
        return centroide

    def obtener_bordes(self,img,lista):


        """ Find edges of cell.

            return 
            list with points of edges 

        """


        lista = sorted(lista, key=itemgetter(1))
        __, max_y = max(lista, key=itemgetter(1))
        __, min_y = min(lista, key=itemgetter(1))
        lim = len(lista)
        lista_puntos = []
        punto = -1

        for i in range(lim):
            x, y = lista[i]
            if y == min_y:
                lista_puntos.append((x, y))
            else:
                punto = i
                lista_puntos.pop()
                break

        for i in range(lim-1,0,-1):
            x, y = lista[i]
            if y == max_y:
                lista_puntos.append((x, y))
            else:
                fin = i
                lista_puntos.append((x, y))
                break
        ant = min_y

        for i in range(punto, fin-1, 1):
            x,y = lista[i]
            if y != ant:
                lista_puntos.append(lista[i-1])
                lista_puntos.append(lista[i])
                ant = y
        return lista_puntos

    def excentricidad(self):


        """ Find cocient between min and max distance.

            return 
            cocient between min distance and max distance 

        """

        centroide = self.find_centroid_cell()
        byw = cv2.imread('byw/imagen' + str(self.__name) + '.png')
        bordes = self.obtener_bordes(byw, self.__col_lista)
        lim = len(bordes)
        dis_centro = []
        for i in range(lim):
            punto = bordes[i]
            dis_centro.append(self.dist(centroide, punto))

        var = np.std(dis_centro)
        maximo = max(dis_centro)
        minimo = min(dis_centro)
        exce = minimo/maximo
        return exce, var

    def principal(self):


        """ Classify the cell.

            return 
            0 if unclassified 
            1 if prophase
            2 if metaphase
            3 if anaphase
            4 if telophase
        """


        sw = self.telofase()
        if sw is False:
            exc, var = self.excentricidad()
            if 0.35 < exc < 0.5:
                self.guardar_imagen('anafase')
                return 3
            elif 0.5 <= exc < 0.65:
                self.guardar_imagen('metafase')
                return 2
            elif exc >= 0.65:
                self.guardar_imagen('profase')
                return 1
            else:
                self.guardar_imagen('sin_clasificar')
                return 0
        else:
            return 4
