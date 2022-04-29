import math

from PIL import Image
import requests
import matplotlib.pyplot as plt
# %config InlineBackend.figure_format = 'retina'
from osgeo import gdal
import ipywidgets as widgets
from IPython.display import display, clear_output
import efficientnetv2, VANFPN, OCR, VANOCR, HROCR, VAN, res18_oar
import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False);

def TestImage(image):
    as_tensor = T.Compose([
            # 将numpy的ndarray转换成形状为(C,H,W)的Tensor格式,且/255归一化到[0,1.0]之间
            T.ToTensor(),
        ])

    return as_tensor(image)

def imgread(fileName):
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
        data = TestImage(data).reshape(1,3,512,512)
    return data


model = VANOCR.van_tiny().cuda()

model.load_state_dict(
    torch.load(r'D:\softwares\PyCharm\pythonProject\TransUNet-main\savemodel\ep290-loss0.423-acc0.888.pth'))

im = r'D:\train\525.tif'

img = imgread(im)




conv_features, enc_attn_weights, dec_attn_weights = [], [], []

hooks = [

    model.ocr.transformer_cross_attention_layers[-1].multihead_attn.register_forward_hook(
        lambda self, input, output: dec_attn_weights.append(output[1])
    ),
]

outputs = model(img.cuda())[-1]

for hook in hooks:
    hook.remove()

dec_attn_weights = dec_attn_weights[0]


h, w = [64, 64]