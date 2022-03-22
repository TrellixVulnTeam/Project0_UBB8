# import matplotlib.pyplot as plt
#
# x=[3,4,5] # [列表]
# y=[2,3,2] # x,y元素个数N应相同
# plt.plot(x,y)
# plt.show()

import os
import cv2
import numpy as np
from osgeo import gdal
#
path = r'D:\train\375.tif'
# labelpath = r'E:\data\classify\512trainlabelgray'
#
# for i in os.listdir(labelpath):
#     if i not in os.listdir(path):
#         print(i)
#         # os.remove(os.path.join(path,i))


# img = cv2.imread(path)
# cv2.imshow('img', img)
def imgread(fileName):
    dataset = gdal.Open(fileName)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    data = dataset.ReadAsArray(0, 0, width, height)
    # 如果是image的话,因为label是单通道
    if (len(data.shape) == 3):
        data = data.swapaxes(1, 0).swapaxes(1, 2)
    return data


def truncated_linear_stretch(image, truncated_value, max_out=255, min_out=0):
    def gray_process(gray):
        truncated_down = np.percentile(gray, truncated_value)
        truncated_up = np.percentile(gray, 100 - truncated_value)
        gray = (gray - truncated_down) / (truncated_up - truncated_down) * (max_out - min_out) + min_out
        gray = np.clip(gray, min_out, max_out)
        gray = np.uint8(gray)
        return gray

    image_stretch = []
    for i in range(image.shape[2]):
        # 只拉伸RGB
        if (i < 3):
            gray = gray_process(image[:, :, i])
        else:
            gray = image[:, :, i]
        image_stretch.append(gray)
    image_stretch = np.array(image_stretch)
    image_stretch = image_stretch.swapaxes(1, 0).swapaxes(1, 2)
    return image_stretch


if __name__ == '__main__':
    img = imgread(path)

    imgstr = truncated_linear_stretch(img,0.5)
    data = cv2.cvtColor(imgstr, cv2.COLOR_BGR2RGB)
    cv2.imshow('img', data)
    cv2.waitKey(0)