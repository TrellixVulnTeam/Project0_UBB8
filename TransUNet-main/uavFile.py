import random
from osgeo import gdal
import numpy as np
import os,shutil
import sys


trainpath = r'F:\lab\data\uavid_v1.5_official_release_image\uavid_val'
files = os.listdir(trainpath)



def gather():
    name = 200
    for i in files:
        trainfile = os.listdir(os.path.join(trainpath,i))
        for j in trainfile:
            if j == 'Images':
                trainimagefile = os.listdir(os.path.join(trainpath, i, j))
                for k in trainimagefile:

                    shutil.copyfile(os.path.join(trainpath, i, j, k), os.path.join(r'F:\lab\data\uavid_v1.5_official_release_image\IMAGETRAIN',f'{name}.png'))
                    name += 1

gather()