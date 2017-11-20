import numpy as np
import cv2
import errno
import os
import shutil
import time


from itertools import *
from os import listdir
from os.path import isfile, join
from operator import itemgetter
from clasificacion_fase import Clasificador

class Preparation:

    def __init__(self, name):
        """ Classify a cell.

        return 0,1,2,3,4 corresponding to 
        unclassified, prophase,metaphase,anaphase,telophase

        Parameters

        name -- name of the sample

        """
        self.__name=name
        self.crear_directorios('byw')
        self.crear_directorios('gris')
        self.crear_directorios('malas')
        self.crear_directorios('buenas')
        self.crear_directorios('profase')
        self.crear_directorios('metafase')
        self.crear_directorios('anafase')
        self.crear_directorios('telofase')
        self.crear_directorios('sin_clasificar')


    def crear_directorios(self, name):
        """ Create a folder who save temporal images.

        Parameters

        name -- name of the folder

        """


        k="/classified/"+name
        if os.path.exists(k):
            shutil.rmtree(os.getcwd() + k)
        try:
            os.makedirs(os.getcwd() + k)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise



    def guardar_imagen(self, img, cont):

        """ Save a image.

            Parameters

            cont -- name of the cell 

        """

        k = 'byw/imagen'+str(cont)+'.png'
        cv2.imwrite(k, img)


    def cargar_imagenes(self):

        """ Load images.

        """

        mypath = 'images/test'
        nombres = []
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        images = np.empty(len(onlyfiles), dtype=object)
        for n in range(0, len(onlyfiles)):
            img = cv2.imread(join(mypath, onlyfiles[n]))
            images[n] = img
            k = onlyfiles[n].split('.')
            nombres.append(k[0])
        
        return images,nombres 


    def guardar_gris(self, img, cont, ruta):

        """ Save image in grayscale.

            Parameters

            img -- image in grayscale
            cont -- name of the celll
            ruta -- path to the folder 

        """

        k = ruta+'/imagen' + str(cont) + '.png'
        cv2.imwrite(k, img)


    def imagen_to_black_and_white(self, img, cont):

        """ Transform image in grayscale to black and white.

            return 
            image in black and white  
            coordinates of minimum color 
            True or False 

            Parameters
            img -- image in grayscale 
            cont -- name of the cell



        """

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.guardar_gris(gray, cont, 'gris')
        sw = self.var_colores(gray, cont)
        minimo, punto = self.min_colors(gray)
        im_bw = self.change_color(gray, minimo)
        return self.clear_edges(im_bw), punto, sw


    def clear_edges(self, img):

        """ Apply medianBlur to image in black and white.

            Parameters

            img -- image to transform 

        """
        return cv2.medianBlur(img, 5)


    def change_color(self, img, minimo):
        
        """ Change color to black an white. 

            return 
            image in black and white

            Parameters
            img -- image to transform
            minimo -- value of minimum color (0-255)
            

        """

        img[img < minimo + 30] = 0
        img[img > 0] = 255
        return img


    def dist(self, x, y):

        """ Find euclidean distance between a pair of tuples.

            return 

            value with euclidean distance

            Parameters
            x -- first tuple
            y -- second tuple  

        """


        x1, y1 = x
        x2, y2 = y
        return np.sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2))


    def min_colors(self, img):

        """ Find the minimum and index of minimum color.

            return 
            minimum color
            index of minimum color


            Parameters
            
            img -- image in black in grayscale 

        """

        minimo = img[img>-1].min()
        m, n = img.shape
        indice = (0, 0)
        for i in range(m):
            for j in range(n):
                if img[i, j] == minimo:
                    indice = (i, j)
        #print(indice)

        return minimo, indice


    def fill_holes(self, img):

        """ Fill holes in black and white image.

            return 
            closing image

            Parameters

            img -- image in black and white  

        """


        kernel = np.ones((2, 2), np.uint8)
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        return closing


    def crear_Diccionario(self, image, num):

        """ Create a dicctionary with distances and coordinates to nucleus' points.

            return 
            list of distances between nucleus's points and coordinate of minimum color
            list of nucleus's points
            True or False

            Parameters

            image -- image to analize
            num -- name of the cell 

        """


        filas, colum, channels = image.shape
        img, punto, sw = self.imagen_to_black_and_white(image, num)
        lista = []
        puntos = []
        if sw is True:
            img = self.fill_holes(img)
            self.guardar_imagen(img, num)
            elemi1 = 0
            for i in product(range(0, filas), range(elemi1, colum)):
                if img[i] == 0:
                    puntos.append(i)
            limite = len(puntos)
            for i in range(0, limite, 1):
                lista.append(self.dist(puntos[i], punto))

        return lista, puntos, sw


    def var_colores(self, img, name):

        """ Find variance of list of colors. 

            return 
            True if var >= 40
            False if var <40

            Parameters
            img -- objective image
            name -- name of the cell 

        """

        var = np.std(img)
        #print('name=',name, 'var=',var)
        if var < 40:
            self.guardar_gris(img, name, 'malas')
            return False
        else:
            self.guardar_gris(img, name, 'buenas')
            return True


    def principal(self, image, nombre):

        """ Find phase of cell. 

            return 
            mean,variance,phase or (-1,-1,-1) if image is unclassified 


            Parameters

            image -- image to classify
            nombre -- name of the cell 

        """

        dista, colores, sw = self.crear_Diccionario(image, nombre)
        if len(colores) > 10 and sw is True:
            classifier = Clasificador(image, nombre, dista, colores)
            med = classifier.media()
            var = classifier.varianza()
            fase = classifier.principal()
            return med, var, fase
        else:
            #print('name=',nombre,'princ no entro')
            return -1, -1, -1


    def show_phases(self, lista, malas):

        """ Show information of sample.

        """


        print("Results:")
        print("Numbers of cells in prophase: {}.".format (lista[1]))
        print("Numbers of cells in metaphase: {}.".format(lista[2]))
        print("Numbers of cells in anaphase: {}.".format(lista[3]))
        print("Numbers of cells in telophase: {}.".format(lista[4]))
        print("Numbers of cells unclassified: {}.".format((lista[0]+malas)))


    def main(self):

        """ Main method who execute all.

            return 
            4 numbers who represent prophase, metaphase,anaphase,telophase,
            unclassified
        """



        imagenes, names = self.cargar_imagenes()
        var = []
        med = []
        fases_vec = [0, 0, 0, 0, 0]
        cont = 0
        malas = 0
        lenIma = len(imagenes)
        for i in range(lenIma):
            media, varianza, fase = self.principal(imagenes[i], names[i])
            if media == -1 and varianza == -1:
                malas = malas+1
            else:
                fases_vec[fase] = fases_vec[fase]+1
                cont = cont+1
                var.append((varianza, names[i]))
                med.append((media, names[i]))
        print('name of image=',self.__name)

        self.show_phases(fases_vec, malas)
        var = sorted(var, key=itemgetter(0))
        med = sorted(med, key=itemgetter(0))
        k = len(var)
        return fases_vec[1], fases_vec[2], fases_vec[3], fases_vec[4], fases_vec[0]

