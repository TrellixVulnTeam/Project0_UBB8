import random
from osgeo import gdal
import numpy as np
import os
import os
import sys

os.environ['PROJ_LIB'] = r'D:\conda\Library\share\proj'
os.environ['GDAL_DATA'] = r'D:\conda\Library\share'
# os.environ['PROJ_LIB'] = os.path.dirname(sys.argv[0])


#  读取tif数据集
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    return dataset


#  保存tif文件函数
def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if dataset != None:
       dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
       dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset

'''
ImagePath 原始影像路径
LabelPath 标签影像路径
IamgeSavePath 原始影像裁剪后保存目录
LabelSavePath 标签影像裁剪后保存目录
CropSize 裁剪尺寸
CutNum 裁剪数量

'''


def RandomCrop(ImagePath, LabelPath, IamgeSavePath, LabelSavePath, CropSize, CutNum):
    for i,j,k in os.walk(ImagePath):
        for a in k:
            print(a)
            dataset_img = readTif(os.path.join(i,a))
            width = dataset_img.RasterXSize
            height = dataset_img.RasterYSize
            proj = dataset_img.GetProjection()
            geotrans = dataset_img.GetGeoTransform()
            img = dataset_img.ReadAsArray(0, 0, width, height)  # 获取哟昂数
            # 据
            dataset_label = readTif(os.path.join(LabelPath,a))
            width1 = dataset_label.RasterXSize
            height1 = dataset_label.RasterYSize
            label = dataset_label.ReadAsArray(0, 0, width1, height1)  # 获取标签数据

            #  获取当前文件夹的文件个数len,并以len+1命名即将裁剪得到的图像
            fileNum = len(os.listdir(IamgeSavePath))
            new_name = fileNum + 1
            while (new_name < CutNum + fileNum + 1):
                #  生成剪切图像的左上角XY坐标
                UpperLeftX = random.randint(0, height - CropSize)
                UpperLeftY = random.randint(0, width - CropSize)
                if (len(img.shape) == 2):
                    imgCrop = img[UpperLeftX: UpperLeftX + CropSize,
                                      UpperLeftY: UpperLeftY + CropSize]
                else:
                    imgCrop = img[:,
                                UpperLeftX: UpperLeftX + CropSize,
                                UpperLeftY: UpperLeftY + CropSize]
                if (len(label.shape) == 2):
                    labelCrop = label[UpperLeftX: UpperLeftX + CropSize,
                                UpperLeftY: UpperLeftY + CropSize]
                else:
                    labelCrop = label[:,
                                UpperLeftX: UpperLeftX + CropSize,
                                UpperLeftY: UpperLeftY + CropSize]
                geotranscrop = ()
                geotranscrop = geotranscrop + (geotrans[0] + UpperLeftX*geotrans[2] + UpperLeftY*geotrans[1],) + (geotrans[1],)+ (geotrans[2],) +(geotrans[3] + UpperLeftX*geotrans[5] + UpperLeftY*geotrans[4],) + (geotrans[4],) + (geotrans[5],)
                # geotranscrop[3] = geotrans[3] + UpperLeftX*geotrans[4] + UpperLeftY*geotrans[5]

                writeTiff(imgCrop, geotranscrop, proj, IamgeSavePath + "/%d.tif" % new_name)
                writeTiff(labelCrop, geotranscrop, proj, LabelSavePath + "/%d.tif" % new_name)
                new_name = new_name + 1
# RandomCrop(r"E:\data\YAMATO\2_Ortho_RGB",
#            r"E:\data\YAMATO\biggrayall",
#            r"F:\lab2021\YAMATO\potsimage",
#            r"F:\lab2021\YAMATO\potslabel",
#            512, 120 )


def TifCrop(TifPath, SavePath, labelpath, labelsavepath, CropSize, RepetitionRate):
    for oo,pp,k in os.walk(TifPath):
        for a in k:
            print(a)
            dataset_img = readTif(os.path.join(oo, a))
            dataset_label = readTif(os.path.join(labelpath, a))
            width = dataset_img.RasterXSize
            height = dataset_img.RasterYSize
            proj = dataset_img.GetProjection()
            geotrans = dataset_img.GetGeoTransform()
            img = dataset_img.ReadAsArray(0, 0, width, height)  # 获取数据
            label = dataset_label.ReadAsArray(0, 0, width, height)
            #  获取当前文件夹的文件个数len,并以len+1命名即将裁剪得到的图像
            new_name = len(os.listdir(SavePath)) + 1
            #  裁剪图片,重复率为RepetitionRate
            lenth = int(CropSize * (1-RepetitionRate))

            # for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
            for i in range(12):
                # for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
                    #  如果图像是单波段
                for j in range(12):

                    cropped = img[:, i*486:i*486 + CropSize, j*486: j*486 + CropSize]
                    cropped_label = label[i*486: i*486 + CropSize, j*486: j*486 + CropSize]
                    #  写图像
                    geotranscrop = ()
                    geotranscrop = geotranscrop + (geotrans[0] + 486*i * geotrans[2] + 486 * j * geotrans[1],) + (
                    geotrans[1],) + (geotrans[2],) + (geotrans[3] + 486 * i * geotrans[5] + 486 * j * geotrans[4],) + (
                                   geotrans[4],) + (geotrans[5],)
                    writeTiff(cropped, geotranscrop, proj, SavePath + "/%d.tif" % new_name)
                    writeTiff(cropped_label, geotranscrop, proj, labelsavepath+"/%d.tif" % new_name)
                     # 文件名 + 1
                    new_name = new_name + 1
             # 向前裁剪最后一列
            for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
                cropped = img[:,
                              int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                              (width - CropSize): width]
                cropped_label = label[
                              int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                              (width - CropSize): width]
                #  写图像
                geotranscrop = ()
                geotranscrop = geotranscrop + (geotrans[0] + lenth * i * geotrans[2] + (width-CropSize) * geotrans[1],) + (
                    geotrans[1],) + (geotrans[2],) + (geotrans[3] + lenth * i * geotrans[5] + (width-CropSize) * geotrans[4],) + (
                                   geotrans[4],) + (geotrans[5],)
                writeTiff(cropped, geotranscrop, proj, SavePath + "/%d.tif" % new_name)
                writeTiff(cropped_label,geotranscrop, proj, labelsavepath + "/%d.tif" % new_name)
                new_name = new_name + 1
             # 向前裁剪最后一行
            for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):

                cropped = img[:,
                              (height - CropSize): height,
                              int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
                cropped_label = label[(height - CropSize): height,
                              int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
                geotranscrop = ()
                geotranscrop = geotranscrop + (geotrans[0] +  (height-CropSize)* geotrans[2] + lenth * j * geotrans[1],) + (
                        geotrans[1],) + (geotrans[2],) + (geotrans[3] + (height-CropSize) * geotrans[5] + lenth * j * geotrans[4],) + (
                                       geotrans[4],) + (geotrans[5],)
                writeTiff(cropped, geotranscrop, proj, SavePath + "/%d.tif" % new_name)
                writeTiff(cropped_label, geotranscrop, proj, labelsavepath + "/%d.tif" % new_name)
                #  文件名 + 1

                new_name = new_name + 1
             # 裁剪右下角

            cropped = img[:,
                          (height - CropSize): height,
                          (width - CropSize): width]
            cropped_label = label[(height - CropSize): height,
                          (width - CropSize): width]
            geotranscrop = ()
            geotranscrop = geotranscrop + (
            geotrans[0] + (height - CropSize) * geotrans[2] + (width-CropSize) * geotrans[1],) + (
                               geotrans[1],) + (geotrans[2],) + (
                           geotrans[3] + (height - CropSize) * geotrans[5] + (width-CropSize) * geotrans[4],) + (
                               geotrans[4],) + (geotrans[5],)
            writeTiff(cropped, geotranscrop, proj, SavePath + "/%d.tif" % new_name)
            writeTiff(cropped_label, geotranscrop, proj, labelsavepath + "/%d.tif" % new_name)
            new_name = new_name + 1

if __name__ == "__main__":
    #  将影像1裁剪为重复率为0.1的256×256的数据集
    TifCrop(r"F:\lab2021\YAMATO\2_Ortho_RGB",
            r"F:\lab2021\YAMATO\POTSimageseq005", r'F:\lab2021\YAMATO\biggrayall', r'F:\lab2021\YAMATO\POTSlabelseq005',512, 0.05)
    # TifCrop(r"Data\data2\label\label.tif",
    #         r"data\train\label1", 256, 0.1)
