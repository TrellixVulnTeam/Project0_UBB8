from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils.data.dataset import Dataset
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import transforms as T
import torch.functional as F
import numpy as np
import requests
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import torch
import torch.nn as nn
from torchvision.models import resnet50
import efficientnetv2, VANFPN, OCR, VANOCR, HROCR, VAN
from pytorch_grad_cam import GradCAM

from osgeo import gdal


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


#
# model = resnet50(pretrained=True)
# model.avgpool=nn.Identity()
# model.fc = nn.Identity()
# net = VANOCR.van_tiny().cuda()
# net.load_state_dict(
#     torch.load(r'D:\softwares\PyCharm\pythonProject\TransUNet-main\model\ep620-loss0.061-acc0.889.pth'))
# target_layerss = net.ocr.head[1]
# target_layers = [model.layer4[-1]]
# cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
# model.train()
# inputs = model(imgread(r'D:\train\377.tif').cuda())
#
# grayscale_cam = cam(input_tensor=inputs, targets=None)



image_url = "https://farm1.staticflickr.com/6/9606553_ccc7518589_z.jpg"
image = np.array(Image.open(r'D:\train\1125.tif'))
rgb_img = np.float32(image) / 255
input_tensor = preprocess_image(rgb_img,
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
# model = deeplabv3_resnet50(pretrained=True, progress=False)
model = VANOCR.van_tiny().cuda()
model.load_state_dict(
    torch.load(r'D:\softwares\PyCharm\pythonProject\TransUNet-main\model\ep620-loss0.061-acc0.889.pth'))
model = model.eval()
if torch.cuda.is_available():
    model = model
    input_tensor = input_tensor.cuda()

# output = model(input_tensor)



# class SegmentationModelOutputWrapper(torch.nn.Module):
#     def __init__(self, model):
#         super(SegmentationModelOutputWrapper, self).__init__()
#         self.model = model
#
#     def forward(self, x):
#         return self.model(x)["out"]


# model = SegmentationModelOutputWrapper(model)
output = model(input_tensor)[-1]
# # print(type(output), output.size())
#
normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()
sem_classes = [
    'impsurf', 'building', 'vegta', 'forest', 'car', 'bg'
]
sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

car_category = sem_class_to_idx["building"]
car_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
car_mask_uint8 = 255 * np.uint8(car_mask == car_category)
car_mask_float = np.float32(car_mask == car_category)
#
# both_images = np.hstack((image, np.repeat(car_mask_uint8[:, :, None], 3, axis=-1)))
# a = Image.fromarray(both_images)
# a.show()
class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[0,self.category, :, :] * self.mask).sum()

# target_layers = [model.model.backbone.layer4]

# target_layers = [model.ocr.head[1]]
# target_layers = [model.ocr.cls_head]
target_layers = [model.ocr.ocr_distri_head.object_context_block.f_up[0]]
# target_layers = [model.fpn.latlayer3]
targets = [SemanticSegmentationTarget(car_category, car_mask_float)]
with GradCAM(model=model,
             target_layers=target_layers,
             use_cuda=torch.cuda.is_available()) as cam:
    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets)[0, :]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

b = Image.fromarray(cam_image)
b.show()