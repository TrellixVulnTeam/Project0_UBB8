import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from sklearn.preprocessing import OneHotEncoder


def get_one_hot(label, N):
    size = list(label.size())
    label = label.view(-1)   # reshape 为向量
    ones = torch.sparse.torch.eye(N)
    ones = ones.index_select(0, label)   # 用上面的办法转为换one hot
    size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
    return ones.view(*size)


def CE_Loss(inputs, target, num_classes=6):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c).cuda()
    temp_target = target.view(-1).cuda()

    CE_loss  = nn.NLLLoss(ignore_index=num_classes)(F.log_softmax(temp_inputs, dim = -1), temp_target)
    return CE_loss


def Dice_loss(inputs, target, beta=1, smooth = 1e-5):
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
    # if h != ht and w != wt:
    #     inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
    #
    # temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1).cuda()
    # temp_target = target.view(n, -1, ct).cuda()
    # tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    # fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    # fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp
    #
    # score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    # dice_loss = 1 - torch.mean(score)
    return dice_loss

def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)
