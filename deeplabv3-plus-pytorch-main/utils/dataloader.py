import os

import cv2
import numpy as np
from PIL import Image
from osgeo import gdal
import random
from torchvision import transforms as T
from torch.utils.data.dataset import Dataset
# from utils.utils import preprocess_input, cvtColor


def imgread(fileName, addNDVI=False):
    dataset = gdal.Open(fileName)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    data = dataset.ReadAsArray(0, 0, width, height)
    # 如果是image的话,因为label是单通道
    if (len(data.shape) == 3):
        # 添加归一化植被指数NDVI特征
        # if (addNDVI):
        #     nir, r = data[3], data[0]
        #     ndvi = (nir - r) / (nir + r + 0.00001) * 1.0
        #     # 和其他波段保持统一,归到0-255,后面的totensor会/255统一归一化
        #     # 统计了所有训练集ndvi的值，最小值为0，最大值很大但是数目很少，所以我们取了98%处的25
        #     ndvi = (ndvi - 0) / (25 - 0) * 255
        #     ndvi = np.clip(ndvi, 0, 255)
        #     data_add_ndvi = np.zeros((5, 256, 256), np.uint8)
        #     data_add_ndvi[0:4] = data
        #     data_add_ndvi[4] = np.uint8(ndvi)
        #     data = data_add_ndvi
        # (C,H,W)->(H,W,C)
        data = data.swapaxes(1, 0).swapaxes(1, 2)
    return data


def split_train_val(image_paths, label_paths, val_index=0, upsample=False):
    # 分隔训练集和验证集
    train_image_paths, train_label_paths, val_image_paths, val_label_paths = [], [], [], []
    for i in range(len(image_paths)):
        # 训练验证4:1,即每5个数据的第val_index个数据为验证集
        if i % 5 == val_index:
            val_image_paths.append(image_paths[i])
            val_label_paths.append(label_paths[i])
        else:
            train_image_paths.append(image_paths[i])
            train_label_paths.append(label_paths[i])
    print("Number of train images: ", len(train_image_paths))
    print("Number of val images: ", len(val_image_paths))
    return train_image_paths, train_label_paths, val_image_paths, val_label_paths

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


def DataAugmentation(image, label, mode):
    if (mode == "train"):
        hor = random.choice([True, False])
        if (hor):
            #  图像水平翻转
            image = np.flip(image, axis=1)
            label = np.flip(label, axis=1)
        ver = random.choice([True, False])
        if (ver):
            #  图像垂直翻转
            image = np.flip(image, axis=0)
            label = np.flip(label, axis=0)
        stretch = random.choice([True, False])
        if (stretch):
            image = truncated_linear_stretch(image, 0.5)
    if (mode == "val"):
        stretch = random.choice([0.8, 1, 2])
        # if(stretch == 'yes'):
        # 0.5%线性拉伸
        image = truncated_linear_stretch(image, stretch)
    return image, label


class DeeplabDataset(Dataset):  #  import torch.utils.data as D   (D.DataSet)

    # def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
    #     super(DeeplabDataset, self).__init__()
    #     self.annotation_lines   = annotation_lines
    #     self.length             = len(annotation_lines)
    #     self.input_shape        = input_shape
    #     self.num_classes        = num_classes
    #     self.train              = train
    #     self.dataset_path       = dataset_path

    def __init__(self, image_paths, label_paths, mode,):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.mode = mode
        self.len = len(image_paths)
        self.as_tensor = T.Compose([
            # 将numpy的ndarray转换成形状为(C,H,W)的Tensor格式,且/255归一化到[0,1.0]之间
            T.ToTensor(),
        ])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image = imgread(self.image_paths[index])
        if self.mode == "train":
            label = imgread(self.label_paths[index], )-1
            image, label = DataAugmentation(image, label, self.mode)
            #  传入一个内存连续的array对象,pytorch要求传入的numpy的array对象必须是内存连续
            image_array = np.ascontiguousarray(image)
            return self.as_tensor(image_array), label.astype(np.int64)
        elif self.mode == "val":
            label = imgread(self.label_paths[index])-1
            # 常规来讲,验证集不需要数据增强,但是这次数据测试集和训练集不同域,为了模拟不同域,验证集也进行数据增强
            image, label = DataAugmentation(image, label, self.mode)
            image_array = np.ascontiguousarray(image)
            return self.as_tensor(image_array), label.astype(np.int64)
        elif self.mode == "test":
            image_stretch = truncated_linear_stretch(image, 0.5)
            image_ndvi = imgread(self.image_paths[index], False)
            nir, r = image_ndvi[ :, 3], image_ndvi[ :, 0]
            # ndvi = (nir - r) / (nir + r + 0.00001) * 1.0
            return self.as_tensor(image), self.as_tensor(image_stretch), self.image_paths[index]
                # , ndvi


    # def __getitem__(self, index):
    #     annotation_line = self.annotation_lines[index]
    #     name            = annotation_line.split()[0]
    #
    #     #-------------------------------#
    #     #   从文件中读取图像
    #     #-------------------------------#
    #     jpg         = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/JPEGImages"), name + ".jpg"))
    #     png         = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/SegmentationClass"), name + ".png"))
    #     #-------------------------------#
    #     #   数据增强
    #     #-------------------------------#
    #     jpg, png    = self.get_random_data(jpg, png, self.input_shape, random = self.train)
    #
    #     jpg         = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2,0,1])
    #     png         = np.array(png)
    #     png[png >= self.num_classes] = self.num_classes
    #     #-------------------------------------------------------#
    #     #   转化成one_hot的形式
    #     #   在这里需要+1是因为voc数据集有些标签具有白边部分
    #     #   我们需要将白边部分进行忽略，+1的目的是方便忽略。
    #     #-------------------------------------------------------#
    #     seg_labels  = np.eye(self.num_classes + 1)[png.reshape([-1])]
    #     seg_labels  = seg_labels.reshape((int(self.input_shape[1]), int(self.input_shape[0]), self.num_classes+1))
    #
    #     return jpg, png, seg_labels

    # def rand(self, a=0, b=1):
    #     return np.random.rand() * (b - a) + a
    #
    # def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
    #     image = cvtColor(image)
    #     label = Image.fromarray(np.array(label))
    #     h, w = input_shape
    #
    #     if not random:
    #         iw, ih  = image.size
    #         scale   = min(w/iw, h/ih)
    #         nw      = int(iw*scale)
    #         nh      = int(ih*scale)
    #
    #         image       = image.resize((nw,nh), Image.BICUBIC)
    #         new_image   = Image.new('RGB', [w, h], (128,128,128))
    #         new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    #
    #         label       = label.resize((nw,nh), Image.NEAREST)
    #         new_label   = Image.new('L', [w, h], (0))
    #         new_label.paste(label, ((w-nw)//2, (h-nh)//2))
    #         return new_image, new_label
    #
    #     # resize image
    #     rand_jit1 = self.rand(1-jitter,1+jitter)
    #     rand_jit2 = self.rand(1-jitter,1+jitter)
    #     new_ar = w/h * rand_jit1/rand_jit2
    #
    #     scale = self.rand(0.25, 2)
    #     if new_ar < 1:
    #         nh = int(scale*h)
    #         nw = int(nh*new_ar)
    #     else:
    #         nw = int(scale*w)
    #         nh = int(nw/new_ar)
    #
    #     image = image.resize((nw,nh), Image.BICUBIC)
    #     label = label.resize((nw,nh), Image.NEAREST)
    #
    #     flip = self.rand()<.5
    #     if flip:
    #         image = image.transpose(Image.FLIP_LEFT_RIGHT)
    #         label = label.transpose(Image.FLIP_LEFT_RIGHT)
    #
    #     # place image
    #     dx = int(self.rand(0, w-nw))
    #     dy = int(self.rand(0, h-nh))
    #     new_image = Image.new('RGB', (w,h), (128,128,128))
    #     new_label = Image.new('L', (w,h), (0))
    #     new_image.paste(image, (dx, dy))
    #     new_label.paste(label, (dx, dy))
    #     image = new_image
    #     label = new_label
    #
    #     # distort image
    #     hue = self.rand(-hue, hue)
    #     sat = self.rand(1, sat) if self.rand()<.5 else 1/self.rand(1, sat)
    #     val = self.rand(1, val) if self.rand()<.5 else 1/self.rand(1, val)
    #     x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
    #     x[..., 0] += hue*360
    #     x[..., 0][x[..., 0]>1] -= 1
    #     x[..., 0][x[..., 0]<0] += 1
    #     x[..., 1] *= sat
    #     x[..., 2] *= val
    #     x[x[:,:, 0]>360, 0] = 360
    #     x[:, :, 1:][x[:, :, 1:]>1] = 1
    #     x[x<0] = 0
    #     image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255
    #     return image_data,label


# DataLoader中collate_fn使用
def deeplab_dataset_collate(batch):
    images      = []
    pngs        = []
    seg_labels  = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images      = np.array(images)
    pngs        = np.array(pngs)
    seg_labels  = np.array(seg_labels)
    return images, pngs, seg_labels

