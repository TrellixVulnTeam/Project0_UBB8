import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import torch
import torch.nn.functional as F
# from ipdb import set_trace
import VANOCR

writer = SummaryWriter(comment='test_your_comment', filename_suffix="_test_your_filename_suffix")

model = VANOCR.van_tiny().cuda()
model.load_state_dict(
    torch.load(r'D:\softwares\PyCharm\pythonProject\TransUNet-main\savemodel\Ours-Pots-140-loss0.200-miou0.850.pth'))
normMean = [0.3390313, 0.36220086, 0.33583698]
normStd = [0.13880502, 0.1363614, 0.14132875]
path_img = r'D:\train\2065.tif'


img_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=normMean,std=normStd)])

img_pil = Image.open(path_img).convert('RGB')
img_tensor = img_transforms(img_pil)
img_tensor.unsqueeze_(0)

fmap_dict = dict()
n = 0

def hook_func(m, i, o):
    key_name = str(m.weight.shape)
    fmap_dict[key_name].append(o)


for name, sub_module in model.named_modules():
    if isinstance(sub_module, nn.Conv2d):
        n += 1
        key_name = str(sub_module.weight.shape)
        fmap_dict.setdefault(key_name, list())
        n1, n2 = name.split(".")
        model._modules[n1]._modules[n2].register_forward_hook(hook_func)

output = model(img_tensor)
print(fmap_dict['torch.Size([128, 64, 3, 3])'][0].shape)

for layer_name, fmap_list in fmap_dict.items():
    fmap = fmap_list[0]
    # print(fmap.shape)
    fmap.transpose_(0, 1)
    # print(fmap.shape)

    nrow = int(np.sqrt(fmap.shape[0]))
    # if layer_name == 'torch.Size([512, 512, 3, 3])':

    fmap = F.interpolate(fmap, size=[512, 512], mode="bilinear")
    fmap_grid = vutils.make_grid(fmap, normalize=True, scale_each=True, nrow=nrow)
    print(type(fmap_grid), fmap_grid.shape)
    writer.add_image(f'feature map in {layer_name}', fmap_grid, global_step=322)
