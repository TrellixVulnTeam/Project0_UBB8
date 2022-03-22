import os
import pandas as pd
import torch
import torchvision
import numpy as np

import cv2
from skimage import io, transform, img_as_float

path = r'E:\data\YAMATO\2_Ortho_RGB'
imagepath = r'E:\data\YAMATO\labelall'
resizepath = r'E:\data\classify\512trainlabelresize'
savepath = r'E:\data\YAMATO\biggrayall'
files = os.listdir(savepath)


# for name in files:
#
image = cv2.imread(r'E:\data\YAMATO\labelall\top_potsdam_6_7_RGB.tif')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
for i  in range(6000):
    for j in range(6000):
            if gray[i, j] == 255 :
                    gray[i, j] = 1
            if gray[i, j] == 29:
                        gray[i, j] = 2
            if gray[i,j] == 179:
                        gray[i,j] = 3
            if gray[i,j] == 150:
                        gray[i,j] = 4
            if gray[i,j] ==226 or gray[i,j]==225:
                        gray[i,j] = 5
            if gray[i,j] ==76:
                        gray[i, j] = 6
print('finish')
cv2.imwrite(os.path.join(r'E:\data\YAMATO\biggrayall\top_potsdam_6_7_RGB.tif'), gray)

# label = cv2.imread(r'E:\data\classify\label.tif')
# gray = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
# for i  in range(4308):
#         for j in range(4850):
#                     if gray[i, j] == 187:
#                         gray[i, j] = 1
#                     if gray[i, j] == 142:
#                         gray[i, j] = 2
#                     if gray[i,j] == 141:
#                         gray[i,j] = 3
#                     if gray[i,j] ==171:
#                         gray[i,j] = 4
#                     if gray[i,j] ==104:
#                         gray[i,j] = 5
#                     if gray[i,j] ==0:
#                         gray[i, j] = 6
# cv2.imwrite(r'E:\data\classify\labelgray.tif', gray)



# #
# for name in files:
#     image = cv2.imread(os.path.join(savepath, name))
#     value = np.unique(image)
#     a = value.max()
#     b = value.min()
#     print(name)
#     if a > 6 or b == 0:
#         # os.remove(os.path.join(savepath, name))
#         # os.remove(os.path.join(imagepath,name))
#         print('%s snuber bigger than 6')

# for name in files:
#     image = cv2.imread(os.path.join(path, name))
#     # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     a,b = image.shape[0],image.shape[1]
#     if a != 512 or b != 512 :
#         image = cv2.resize(image, (512, 512))
#         cv2.imwrite(os.path.join(resizepath, name),image)
#     else:
#         cv2.imwrite(os.path.join(resizepath, name),image)
# list = []
# for name in files:
#
#     label = os.listdir(imagepath)
#     if name in label:
#         list.append(name)
# print(len(list))