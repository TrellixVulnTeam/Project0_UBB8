from os.path import join

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from nets.deeplabv3_training import get_one_hot
from utils.utils import resize_image, cvtColor, preprocess_input
from torchsummary import summary

def f_score(inputs, target, beta=1, smooth = 1e-5, threhold = 0.5):
    n, c, h, w = inputs.size()
    # arra = target.numpy()
    # arra1 = arra[0, :, :] *50
    # image = Image.fromarray(np.uint8(arra1))
    # image = cvtColor(image)
    # arrai
    # image.show()
    target = get_one_hot(target, c)
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct).cuda()

    #--------------------------------------------#
    #   计算dice系数
    #--------------------------------------------#
    temp_inputs = torch.gt(temp_inputs, threhold).float()
    tp = torch.sum(temp_target * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs, axis=[0,1]) - tp
    fn = torch.sum(temp_target, axis=[0,1]) - tp
    tn = 512*512*8 - (tp+fp+fn)

    inputss = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1).cuda()
    inp = inputss > threhold
    ta = target.view(n, -1, ct) == torch.max(target.view(n, -1, ct))
    ta = ta.cuda()
    inp = inp.cuda()

    tpp = torch.sum((inp == 1) & (ta == 1) == 1 ,axis = [0,1])
    fnn = torch.sum(((inp == 0) & (ta == 1))==1,axis = [0,1])
    tnn = torch.sum(((inp == 0) & (ta == 0))==1,axis =[0,1])
    fpp = torch.sum(((inp == 1) & (ta == 0))==1,axis = [0,1])

    se = float(torch.sum(tpp)) / (float(torch.sum(tpp + fnn)) + 1e-5)
    pc = float(torch.sum(tpp)) / (float(torch.sum(tpp + fpp)) + 1e-5)

    F1 = 2 * se * pc / (se + pc + 1e-5)
    oa = torch.mean((tp+tn)/(tp+fn+fp+tn))
    oa2 = (tpp+tnn)/(tpp+fnn+fpp+tnn)

    size = target.size(0) * target.size(1) * target.size(2)*target.size(3)
    corr = torch.sum(inp == ta)
    oaa = float(corr)/float(size)
    acc = torch.mean(tp/(tp+fp+smooth))

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    iou = tp/(fp + fn + tp + smooth)
    miou = torch.mean(iou)
    # return loss.sum(dim=0).mean()
    return score, miou.item(), acc.item(), oa.item()


def cal_mIou(pred,mask,c = 6):
    iou_result = []
    mask = get_one_hot(mask, c).cuda()
    for idx in range(c):
        p = (mask == idx).int().reshape(-1)
        t = (pred == idx).int().reshape(-1)
        uion = p.sum() + t.sum()
        overlap = (p * t).sum()
        #  0.0001防止除零
        iou = 2 * overlap / (uion + 0.0001)
        iou_result.append(iou.abs().data.cpu().numpy())
    return np.stack(iou_result)


# 设标签宽W，长H
def fast_hist(a, b, n):
    #--------------------------------------------------------------------------------#
    #   a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的预测结果，形状(H×W,)
    #--------------------------------------------------------------------------------#
    k = (a >= 0) & (a < n)
    #--------------------------------------------------------------------------------#
    #   np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    #   返回中，写对角线上的为分类正确的像素点
    #--------------------------------------------------------------------------------#
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)  

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1) 

def per_class_PA(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1) 

def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes):  
    print('Num classes', num_classes)  
    #-----------------------------------------#
    #   创建一个全是0的矩阵，是一个混淆矩阵
    #-----------------------------------------#
    hist = np.zeros((num_classes, num_classes))
    
    #------------------------------------------------#
    #   获得验证集标签路径列表，方便直接读取
    #   获得验证集图像分割结果路径列表，方便直接读取
    #------------------------------------------------#
    gt_imgs     = [join(gt_dir, x + ".png") for x in png_name_list]  
    pred_imgs   = [join(pred_dir, x + ".png") for x in png_name_list]  

    #------------------------------------------------#
    #   读取每一个（图片-标签）对
    #------------------------------------------------#
    for ind in range(len(gt_imgs)): 
        #------------------------------------------------#
        #   读取一张图像分割结果，转化成numpy数组
        #------------------------------------------------#
        pred = np.array(Image.open(pred_imgs[ind]))  
        #------------------------------------------------#
        #   读取一张对应的标签，转化成numpy数组
        #------------------------------------------------#
        label = np.array(Image.open(gt_imgs[ind]))  

        # 如果图像分割结果与标签的大小不一样，这张图片就不计算
        if len(label.flatten()) != len(pred.flatten()):  
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        #------------------------------------------------#
        #   对一张图片计算21×21的hist矩阵，并累加
        #------------------------------------------------#
        hist += fast_hist(label.flatten(), pred.flatten(),num_classes)  
        # 每计算10张就输出一下目前已计算的图片中所有类别平均的mIoU值
        if ind > 0 and ind % 10 == 0:  
            print('{:d} / {:d}: mIou-{:0.2f}; mPA-{:0.2f}'.format(ind, len(gt_imgs),
                                                    100 * np.nanmean(per_class_iu(hist)),
                                                    100 * np.nanmean(per_class_PA(hist))))
    #------------------------------------------------#
    #   计算所有验证集图片的逐类别mIoU值
    #------------------------------------------------#
    mIoUs   = per_class_iu(hist)
    mPA     = per_class_PA(hist)
    #------------------------------------------------#
    #   逐类别输出一下mIoU值
    #------------------------------------------------#
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\tmIou-' + str(round(mIoUs[ind_class] * 100, 2)) + '; mPA-' + str(round(mPA[ind_class] * 100, 2)))

    #-----------------------------------------------------------------#
    #   在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
    #-----------------------------------------------------------------#
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)) + '; mPA: ' + str(round(np.nanmean(mPA) * 100, 2)))  
    return mIoUs
