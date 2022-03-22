from osgeo import gdal
import numpy as np
import os
from PIL import Image
import sys


def readTif (fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    return dataset


def delimage(imagepath,labelpath):
    # NANimage = np.array(Image.open(image))
    filenames = os.listdir(labelpath)
    for k in filenames:


        # image1 = np.array(Image.open(os.path.join(imagepath, i)))
    #     label =  np.array(Image.open(os.path.join(labelpath,i)))
    #     if(np.array_equal(NANimage, label)):
    #         os.remove(os.path.join(imagepath, i))
    #         os.remove(os.path.join(labelpath, i))

        image1 =Image.open(os.path.join(labelpath, k))

        pixel_dict = []
        rows, cols = image1.size
        for i in range(cols):
            n = 0
            color_list = []
            color_max = []
            for j in range(rows):

                pixel = image1.getpixel((j,i))
                if pixel == (0, 0, 0):
                    n = n + 1
                    if n > 32768:
                        os.remove(os.path.join(labelpath, k))
                        os.remove(os.path.join(imagepath, k))
                    break
            else:
                continue
            break






delimage(r'E:\data\Sentinal2A\gztest\cuttest\imagetest', r'E:\data\Sentinal2A\gztest\cuttest\labeltest')

