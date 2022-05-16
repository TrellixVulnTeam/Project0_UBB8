# import argparse
# import logging
import os
import random
# import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
# from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torchvision import transforms as T
from osgeo import gdal
import time
import torch.nn.functional as F
from PIL import Image
from PIL import ImageEnhance
# import imgaug.augmenters as iaa
# import visdom
from itertools import filterfalse as ifilterfalse
from torch.autograd import Variable
from lova import lovasz_softmax
# vis = visdom.Visdom()


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


def RandomCrop(image,label):

        height, width, _ = image.shape

        for _ in range(5):
                    current_image = image

                    # w = random.uniform(0.3 * width, width)
                    # h = random.uniform(0.3 * height, height)
                    #
                    # # aspect ratio constraint b/t .5 & 2
                    # if h / w < 0.5 or h / w > 2:
                    #     continue
                    #
                    w = 224
                    h = 224

                    left = random.uniform(0, width - w)
                    top = random.uniform(0, height - h)

                    rect = np.array([int(left), int(top), int(left + w), int(top + h)])
                    current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]
                    current_labels = label[rect[1]:rect[3], rect[0]:rect[2]]
                    # img = Image.fromarray(current_image)
                    # a = current_labels*20
                    # lab = Image.fromarray(a)
                    # img.show()
                    # lab.show()
                    yield current_image, current_labels


# if __name__ == '__main__':
#     aug = RandomCrop()
#     a = aug.crop(imgread(r'D:\train\145.tif'), imgread(r'D:\labelgray\145.tif'))


def randomColor(image): #随机颜色
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """
    image = Image.fromarray(image)
    # image = Image.open(os.path.join(root_path, img_name))
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    sharpness_image = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)
    return np.array(sharpness_image)


def DataAugmentation(image, label, mode):
    if (mode == "train"):
        # pass
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

        rot90 = random.choice([True,False])
        if(rot90):
            image = np.rot90(image)
            label = np.rot90(label)
        # radcrop = RandomCrop()
        stretch = random.choice([True, False])
        if (stretch):
            image = truncated_linear_stretch(image, 0.5)
        image = randomColor(image)
        # image = RandomCrop(image, label)

        # ColorAug = random.choice([True, False])
        # if(ColorAug):

    # if (mode == "val"):
    #     stretch = random.choice([0.8, 1, 2])
    #     # if(stretch == 'yes'):
    #     # 0.5%线性拉伸
    #     image = truncated_linear_stretch(image, stretch)
    return image, label


# class FocalLoss(nn.Module):
#
#     def __init__(self, alpha=.25, gamma=2):
#         super(FocalLoss, self).__init__()
#         self.alpha = torch.tensor([alpha, 1 - alpha]).cuda()
#         self.gamma = gamma
#
#     def forward(self, inputs, targets):
#         # inputs = inputs.permute(1,0,2,3).contiguous().view(6,-1)
#         targets = get_one_hot(targets,6).permute(0, 3, 1, 2)
#         BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none').contiguous().view(-1)
#         targets = targets.type(torch.long)
#         at = self.alpha.gather(0, targets.data.contiguous().view(-1))
#         pt = torch.exp(-BCE_loss)
#         F_loss = at * (1 - pt) ** self.gamma * BCE_loss
#         return F_loss.mean()

# def isnan(x):
#     return x != x


# def mean(l, ignore_nan=False, empty=0):
#     """
#     nanmean compatible with generators.
#     """
#     l = iter(l)
#     if ignore_nan:
#         l = ifilterfalse(isnan, l)
#     try:
#         n = 1
#         acc = next(l)
#     except StopIteration:
#         if empty == 'raise':
#             raise ValueError('Empty mean')
#         return empty
#     for n, v in enumerate(l, 2):
#         acc += v
#     if n == 1:
#         return acc
#     return acc / n


def flatten_probas(probas, labels, ignore=None):
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss


def CE_Loss(inputs, target, num_classes=6):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c).cuda()
    temp_target = target.view(-1).cuda()

    CE_loss = nn.NLLLoss(ignore_index=num_classes)(F.log_softmax(temp_inputs, dim=-1), temp_target)
    return CE_loss


def get_one_hot(label, N):
    size = list(label.size())
    label1 = label.view(-1)  # reshape 为向量
    ones = torch.sparse.torch.eye(N).cuda()
    ones = ones.index_select(0, label1)   # 用上面的办法转为换one hot
    size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
    return ones.view(*size)


class OhemCELoss(nn.Module):

    def __init__(self, thresh=0.5, ignore_lb=255):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, labels):
        n_min = labels[labels != self.ignore_lb].numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)


def Dice_loss(inputs, target, beta=1, smooth=1e-5):
    n, c, h, w = inputs.size()
    target = get_one_hot(target, c)
    nt, ht, wt, ct = target.size()

    assert target.size() == target.size(), "the size of predict and target must be equal."
    num = target.size(0)

    pre = torch.sigmoid(target).view(num, -1)
    tar = target.view(num, -1)

    intersection = (pre * tar).sum(-1).sum()  # 利用预测值与标签相乘当作交集
    union = (pre + tar).sum(-1).sum()

    dice_loss = 1 - 2 * (intersection + smooth) / (union + smooth)

    return dice_loss


def f_score(inputs, target, beta=1, smooth=1e-5, threhold=0.5):
    n, c, h, w = inputs.size()
    # arra = target.numpy()
    # arra1 = arra[0, :, :] *50
    # image = Image.fromarray(np.uint8(arra1))
    # image = cvtColor(image)
    # arrai
    # image.show()
    target_oh = get_one_hot(target, c)  # [:, :, :, :5]
    target_v = target.view(n, -1)
    nt, ht, wt, ct = target_oh.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target_oh.view(n, -1, ct).cuda()

    # temp_inputs = torch.gt(temp_inputs, threhold).float()
    # temp_squ = torch.topk(temp_inputs, 1)[1].squeeze(2)

    temp_squ = torch.argmax(temp_inputs, -1)
    diff = torch.sum((temp_squ - target_v) == 0)
    size_squeeze = temp_squ.size(0)*temp_squ.size(1)
    oa = diff/size_squeeze

    temp_inputs = get_one_hot(temp_squ, 6)
    tp = torch.sum(temp_target * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target, axis=[0, 1]) - tp
    tn = 512 * 512 * 2 - (tp + fp + fn)
    # tn1 = 512 * 512 * 6 - (tp + fp + fn)

    # inputss = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1).cuda()
    # inp = inputss > threhold
    # ta = target_oh.view(n, -1, ct) == torch.max(target_oh.view(n, -1, ct))
    # ta = ta.cuda()
    # inp = inp.cuda()

    # tpp = torch.sum((inp == 1) & (ta == 1) == 1, axis=[0, 1])
    # fnn = torch.sum(((inp == 0) & (ta == 1)) == 1, axis=[0, 1])
    # tnn = torch.sum(((inp == 0) & (ta == 0)) == 1, axis=[0, 1])
    # fpp = torch.sum(((inp == 1) & (ta == 0)) == 1, axis=[0, 1])

    # se = float(torch.sum(tp)) / (float(torch.sum(tp + fn)) + 1e-5)
    # pc = float(torch.sum(tp)) / (float(torch.sum(tp + fp)) + 1e-5)
    # se =(float(tp) / (float(tp + fn) + 1e-5))
    # F1 = 2 * se * pc / (se + pc + 1e-5)
    # f1 = (2*tp/(2*tp+fp+fn+smooth))[:-1]
    # mF1 = f1[f1 != 0].mean()
    # oa = torch.mean((tp + tn) / (tp + fn + fp + tn))
    # oa2 = (tpp + tnn) / (tpp + fnn + fpp + tnn)
    # oa2 = torch.mean(oa2)


    # size = target_oh.size(0) * target_oh.size(1) * target_oh.size(2) * target_oh.size(3)
    # corr = torch.sum(inp == ta)
    # oaa = float(corr) / float(size)
    # acc = torch.mean(tp / (tp + fp + smooth))
    # recall = tp/(tp+fn + smooth)
    # ff = 2/(1/acc +1/recall)
    # score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    # score = torch.mean(score)
    # iou = tp / (fp + fn + tp + smooth)
    # iou_ground = iou[-1]
    # iou_noground = iou[:-1]
    # miou = torch.mean(iou_noground[iou_noground != 0])
    # return loss.sum(dim=0).mean()
    # return miou.item(), acc.mean().item(), oa.item(), mF1.item(), f1, iou_ground.item(),
    return oa.item(), tp, fp, fn

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def split_train_val(image_paths, label_paths, val_index=0):
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


def adjust_learning_rate(optimizer, base_lr, max_iters, cur_iters, warmup_iter=None, power=0.9):
    if warmup_iter is not None and cur_iters < warmup_iter:
        lr = base_lr * cur_iters / (warmup_iter + 1e-8)
    elif warmup_iter is not None:
        lr = base_lr*((1-float(cur_iters - warmup_iter) / (max_iters - warmup_iter))**(power))
    else:
        lr = base_lr * ((1 - float(cur_iters / max_iters)) ** (power))
    optimizer.param_groups[0]['lr'] = lr
    return lr


def adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr, power):
    lr = base_lr * (1-epoch/num_epochs)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr




def aug_crop(train_image_paths, train_label_paths,t_number, width, height):
    images = {}
    labels = {}
    n = 0
    for i in range(t_number):
        image_big = imgread(train_image_paths[i])
        label_big = imgread(train_label_paths[i])
        n += 1
        # print(n)
        for _ in range(120):
                w = 512
                h = 512
                left = random.uniform(0, width - w)
                # if _ == 1 or _ ==2:
                    # print(left)
                top = random.uniform(0, height - h)
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])
                current_image = image_big[rect[1]:rect[3], rect[0]:rect[2], :]
                current_labels = label_big[rect[1]:rect[3], rect[0]:rect[2]]
                # ll.append(rect)
                # images[n] = current_image
                # labels[n] = current_labels
                images[120 * i + _] = current_image
                labels[120 * i + _] = current_labels
                # images.append(current_image)
                # labels.append(current_labels)
    return images, labels


class TUDataset(Dataset):  #  import torch.utils.data as D   (D.DataSet)
    def __init__(self, image_paths, label_paths, mode,):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.mode = mode
        # self.aname = aname
        self.len = len(image_paths)
        # self.len1 = len(image_paths)
        self.as_tensor = T.Compose([
            # 将numpy的ndarray转换成形状为(C,H,W)的Tensor格式,且/255归一化到[0,1.0]之间
            T.ToTensor(),
        ])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # print(index)
        image = imgread(self.image_paths[index])
        if self.mode == "train":
            # image = self.image_paths[index]
            label = imgread(self.label_paths[index])-1
            # label = imgread(self.label_paths[index], )-1
            image, label = DataAugmentation(image, label, self.mode)
            #  传入一个内存连续的array对象,pytorch要求传入的numpy的array对象必须是内存连续
            image_array = np.ascontiguousarray(image)
            return self.as_tensor(image_array), label.astype(np.int64)
        elif self.mode == "val":
            # image = self.image_idx[index]
            label = imgread(self.label_paths[index])-1
            # label = imgread(self.label_paths[index])-1
            # # 常规来讲,验证集不需要数据增强,但是这次数据测试集和训练集不同域,为了模拟不同域,验证集也进行数据增强
            # image, label = DataAugmentation(image, label, self.mode)
            image = truncated_linear_stretch(image, 0.5)
            image_array = np.ascontiguousarray(image)
            return self.as_tensor(image_array), label.astype(np.int64)
        elif self.mode == "test":
            image_stretch = truncated_linear_stretch(image, 0.5)
            # image_ndvi = imgread(self.image_paths[index], False)
            # nir, r = image_ndvi[ :, 3], image_ndvi[ :, 0]
            return self.as_tensor(image), self.as_tensor(image_stretch), self.image_paths[index]


class TUDataset2(Dataset):  #  import torch.utils.data as D   (D.DataSet)
    def __init__(self, image_paths, label_paths, mode,):
        # super().__init__()
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.mode = mode
        # self.aname = aname
        self.len = len(image_paths)
        # self.len1 = len(image_paths)
        self.as_tensor = T.Compose([
            # 将numpy的ndarray转换成形状为(C,H,W)的Tensor格式,且/255归一化到[0,1.0]之间
            T.ToTensor(),
        ])

    def __len__(self):
        # files = os.listdir(self.image_paths)
        # lenn = len(files)
        return self.len

    def __getitem__(self, index):
        # image = imgread(self.image_paths[index])

        if self.mode == "train":
            image = self.image_paths[index]
            label = self.label_paths[index]-1
            # label = imgread(self.label_paths[index], )-1
            image, label = DataAugmentation(image, label, self.mode)
            #  传入一个内存连续的array对象,pytorch要求传入的numpy的array对象必须是内存连续
            image_array = np.ascontiguousarray(image)

            return self.as_tensor(image_array), label.astype(np.int64)
        elif self.mode == "val":

            image = self.image_paths[index]
            label = self.label_paths[index]-1
            # label = imgread(self.label_paths[index])-1
            # # 常规来讲,验证集不需要数据增强,但是这次数据测试集和训练集不同域,为了模拟不同域,验证集也进行数据增强
            # # image, label = DataAugmentation(image, label, self.mode)
            image_array = np.ascontiguousarray(image)
            return self.as_tensor(image_array), label.astype(np.int64)


if __name__ == '__main__':
    image_path = r'D:\train'
    import glob
    from torch.autograd import Variable
    # import visdom
    # vis = visdom.Visdom()
    # vis.text('Hello, world!')
    train_num = len(os.listdir(image_path))
    val_num = train_num // 5
    epoch_step = train_num // 6
    epoch_step_val = val_num // 6
    image_paths = glob.glob(r'E:\data\YAMATO\2_Ortho_RGB\*tif')
    label_paths = glob.glob(r'E:\data\YAMATO\biggrayall\*tif')
    train_image_paths, train_label_paths, val_image_paths, val_label_paths = split_train_val(image_paths,
                                                                                             label_paths,
                                                                                             0)
    train_image_paths1, train_label_paths1 = aug_crop(train_image_paths, train_label_paths, 29, 6000, 6000)
    db_train = TUDataset2(train_image_paths1, train_label_paths1, mode='train')
    trainloader = DataLoader(db_train, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    for epoch in range(5):
        # L = random.sample(range(0, 140), 140)
        # n = L[epoch]


        # print(n)
        for i, data in enumerate(trainloader):
            print(i)
            iinput,label = data
            iinput = torch.as_tensor(iinput).cuda()
            label = torch.as_tensor(label).cuda()
            inputs, labels = Variable(iinput), Variable(label)
            print("epoch：", epoch, "的第", i, "个inputs", inputs.data.size(), "labels", labels.data)
            if i == 3:
                break




def trainn(num_classes, model, train_image_paths, val_image_paths, epoch_step, epoch_step_val, train_label_paths,
           val_label_paths, epoch, optimizer,batch_size):
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2,)
    # lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=3e-5, max_lr=3e-3, step_size_up=736, step_size_down=736,
    #                                               mode='triangular2', gamma=1.0, scale_fn=None, scale_mode='cycle',
    #                                               cycle_momentum=False,
    #                                               last_epoch=-1)

    lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False)

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    # iterator = tqdm(range(epoch))
    # lr = args.base_lr
    model = model.cuda()
    # model.load_state_dict(torch.load(r'D:\softwares\PyCharm\pythonProject\TransUNet-main\savemodel\ep010-loss0.000-acc0.000.pth'))

    a = torch.load(r'D:\weight\van_tiny_754.pth.tar')['state_dict']
    # a = torch.load(r'D:\weight\van_small_811.pth.tar')['state_dict']
    # a = torch.load(r'D:\download\CS_scenes_60000-r2-79.7.pth')
    model2_dict = model.state_dict()
    state_dict = {k: v for k, v in a.items() if k in model2_dict.keys()}
    model2_dict.update(state_dict)
    model.load_state_dict(model2_dict)

    iter_num = 0
    db_train = TUDataset(train_image_paths, train_label_paths, mode='train')
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    db_train_val = TUDataset(val_image_paths, val_label_paths, mode='val')
    valloader = DataLoader(db_train_val, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    # lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3, verbose=True)
    a = time.time()

    lrlist = []
    mioulist = []

    for epoch_num in range(epoch):
        vvfp = 0
        vvtp = 0
        vvoa = 0
        vvloss = 0
        vvfn = 0
        vvf1 = 0
        vvgiou =0

        ttacc = 0
        ttmiou = 0
        ttoa = 0
        ttloss = 0
        tttp = 0
        ttfp = 0
        ttoa = 0
        ttfn = 0


        # model.train()
        trep = epoch_step - epoch_step_val
        # train_image_paths, train_label_paths = aug_crop(train_image_paths, train_label_paths,512,512)
        # train_image_paths1, train_label_paths1 = aug_crop(train_image_paths, train_label_paths, 29, 6000, 6000)


        focal_loss1 = FocalLoss()
        with tqdm(total=trep, desc=f'Epoch {epoch_num + 1}/{epoch}', postfix=dict, mininterval=0.3, colour='cyan') as pbar:
            # if __name__ == '__main__':
                model.train()

                for i, sampled in enumerate(trainloader):
                    image_batch, label_batch = sampled
                    del sampled
                    image_batch = image_batch.cuda()
                    label_batch = label_batch.cuda()
                    outputs = model(image_batch)
                    lossp = []
                    weightt = [0.1, 0.2, 0.3, 0.4, 1]
                    k = 1

                    for j in outputs[:-1]:
                        los = dice_loss(j, label_batch,  softmax=True)
                        # los = lovasz_softmax(F.softmax(j, dim=1), label_batch)
                        los2 = (focal_loss1(j, label_batch))  # +/2
                        # if k == 4:
                        auxloss = ((los+los2)/2)
                                  # * (k*0.05)
                        lossp.append(auxloss)
                        # else:
                        #     loss.append((los + los2)*(j+1)*0.1)
                        k += 1
                    lossmain = 0.5*dice_loss(outputs[-1], label_batch, softmax=True)+0.5*focal_loss1(outputs[-1], label_batch)
                    # lossmain = 0.5*lovasz_softmax(F.softmax(outputs[-1], dim=1), label_batch)+0.5*focal_loss1(outputs[-1], label_batch)
                    lossp.append(lossmain)
                    loss = sum(lossp)

                    # aux = outputs[0]
                    # outputs = outputs[1]
                    # lossaux = (focal_loss1(aux, label_batch) + dice_loss(aux, label_batch)) / 2

                    # loss_dice = dice_loss(outputs, label_batch[:].long(),softmax=True)
                    # loss_ce = ce_loss(outputs, label_batch)
                    # # loss_ce = focal_loss1(outputs, label_batch)
                    # loss = 0.5 * loss_ce + 0.5 * loss_dice

                    # los = loss + lossaux

                    #
                    # bisenet loss
                    # outputs, *logits_aux = model(image_batch)
                    # loss_pre = OhemCELoss(0.7)(outputs, label_batch)
                    # loss_aux = [crit(lgt, label_batch) for crit, lgt in zip([OhemCELoss(0.7) for _ in range(4)], logits_aux)]
                    # loss = loss_pre + sum(loss_aux)
                    # lossaux = (ce_loss(aux, label_batch[:].long())+dice_loss(aux, label_batch, softmax=True))/2
                    # loss_ce = ce_loss(outputs, label_batch[:].long())
                    # loss_dice = focal_loss1(outputs, label_batch, softmax=True)
                    # los = dice_loss(aux, label_batch, softmax=True) +loss_dice
                    # los = lovasz_softmax(aux, label_batch) + loss_ce
                    # los = ce_loss(aux, label_batch) + ce_loss(outputs, label_batch)

                    outputs = outputs[-1]
                    with torch.no_grad():
                        # tmiou, tacc, toa, tf1, _, _ = f_score(outputs, label_batch)
                        oa, tp, fp, fn = f_score(outputs, label_batch)
                    # base_lr = 0.006
                    # lr = adjust_learning_rate(optimizer,
                    #                           base_lr,
                    #                           epoch * epoch_step,
                    #                           i + epoch_num * epoch_step,
                    #                           # args2.warm_epochs * lenth_iter
                    #                           )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    tttp += tp
                    ttfp += fp
                    ttoa += oa
                    ttfn += fn
                    vvloss += loss.item()
                    iou = tttp / (ttfp + ttfn + tttp + 1e-5)
                    f1 = ((2 * tttp) / (2 * tttp + ttfp + ttfn + 1e-5))
                    ttloss += loss.item()
                    lr = get_lr(optimizer)

                    # lr_ = lr * (1.0 - iter_num / max_iterations) ** 0.9
                    # for param_group in optimizer.param_groups:
                    # param_group['lr'] = lr_
                    # x = torch.arange(1, 300, 0.01)
                    # x = torch.tensor(i*x[epoch_num+1]).reshape(1)
                    # vis.line(X=x, Y=np.array(ttloss / (i+1)).reshape(1), update='append', win='polynomial', opts={'title': 'loss'})

                    iter_num = iter_num + 1
                    # lr_scheduler.step()
                    pbar.set_postfix(**{'lr': lr,
                                        'oa': ttoa / (i + 1),
                                        # 'iou': np.array(iou.cpu()),
                                        'mIou': iou[:-1].mean().item(),
                                        'loss': ttloss / (i+1),
                                        'f1': f1[-1].mean().item()
                                        # 'f1': np.array(f1.cpu()),
                                        })

                    pbar.update(1)
        print(f'f1:{f1} ||  iou: {iou} || oa: {ttoa / (i + 1):.3f}  || loss: {ttloss / (i + 1):.3f}')


        if (epoch_num+1) % 10 == 0:
        # if epoch_num == 0 or 1:
        # while True:
        #     val_image_paths, val_label_paths = aug_crop(val_image_paths, val_label_paths)
        #     val_image_paths1, val_label_paths1 = aug_crop( val_image_paths, val_label_paths, 8, 6000, 6000)

            model.eval()
            with tqdm(total=epoch_step_val, desc=f'Epoch {epoch_num + 1}/{epoch}', postfix=dict, mininterval=0.3) as pbar:
                for j, vsampled in enumerate(valloader):
                    if j >= epoch_step_val:
                        break
                    vimage_batch, vlabel_batch = vsampled
                    vimage_batch = vimage_batch.cuda()
                    vlabel_batch = vlabel_batch.cuda()
                    with torch.no_grad():
                        outputss = model(vimage_batch)
                        outputss = outputss[-1]
                        vloss_ce = ce_loss(outputss, vlabel_batch[:].long())
                        vloss_dice = dice_loss(outputss, vlabel_batch, softmax=True)
                        vloss = 0.5 * vloss_ce + 0.5 * vloss_dice
                        #outputss, *logits_aux = model(vimage_batch)
                        # loss_pre = OhemCELoss(0.7)(outputss, vlabel_batch)
                        # loss_aux = [crit(lgt, vlabel_batch) for crit, lgt in
                        #             zip([OhemCELoss(0.7) for _ in range(4)], logits_aux)]
                        # vloss = loss_pre
                        # vmiou, vacc, voa, vmf1, vf1, vgiou = f_score(outputss, vlabel_batch)
                        voa, tp, fp, fn = f_score(outputss, vlabel_batch)
                        vvtp += tp
                        vvfp += fp
                        vvoa += voa
                        vvfn += fn
                        vvloss += vloss
                        iou = vvtp / (vvfp + vvfn + vvtp + 1e-5)
                        f1 = ((2 * vvtp) / (2 * vvtp + vvfp + vvfn + 1e-5))
                    pbar.set_postfix(**{'lr': get_lr(optimizer),
                                        'oa': vvoa / (j+1),
                                        # 'iou': iou,
                                        'mIou': iou[:-1].mean().item(),
                                        # 'f1': f1,
                                        'mf1': f1[:-1].mean().item(),
                                        })
                    pbar.update(1)
                print(f'f1:{f1} ||  iou: {iou} || oa: {vvoa / (i + 1):.3f}  || loss: {vvloss / (i + 1):.3f}')
            if 140 >= epoch_num >= 120 or epoch_num >= 280:
                torch.save(model.state_dict(), 'savemodel/ep%03d-loss%.3f-mIou%.3f.pth' % ((epoch_num + 1), vvloss / (epoch_step_val),  iou[:-1].mean().item()))
        lr_scheduler.step()

    print((time.time() - a)/60)
