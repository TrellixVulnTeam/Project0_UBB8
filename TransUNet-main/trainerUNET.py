import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torchvision import transforms as T
from osgeo import gdal
import time
import torch.nn.functional as F
from .UNET import UNET

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


class TUDataset(Dataset):  #  import torch.utils.data as D   (D.DataSet)

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
        # files = os.listdir(self.image_paths)
        # lenn = len(files)
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


def trainer_synapse(args, model, snapshot_path, image_paths, label_paths):
    # def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    # db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
    #                            transform=transforms.Compose(
    #                                [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    db_train =TUDataset(image_paths, label_paths,  mode='train')


    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,)
                             # worker_init_fn=worker_init_fn)
    # DataLoader(db_train, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    # if args.n_gpu > 1:
    #     model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch
            # ['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"

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
    label = label.view(-1)  # reshape 为向量
    ones = torch.sparse.torch.eye(N).cuda()
    ones = ones.index_select(0, label)   # 用上面的办法转为换one hot
    size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
    return ones.view(*size)


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
    target_oh = get_one_hot(target, c)
    target_v = target.view(2, 262144)
    nt, ht, wt, ct = target_oh.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target_oh.view(n, -1, ct).cuda()

    temp_inputs = torch.gt(temp_inputs, threhold).float()
    temp_squ = torch.topk(temp_inputs, 1)[1].squeeze(2)
    diff = torch.sum((temp_squ - target_v) == 0)
    size_squeeze = temp_squ.size(0)*temp_squ.size(1)
    oa = diff/size_squeeze


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

    # se = float(torch.sum(tpp)) / (float(torch.sum(tpp + fnn)) + 1e-5)
    # pc = float(torch.sum(tpp)) / (float(torch.sum(tpp + fpp)) + 1e-5)

    # F1 = 2 * se * pc / (se + pc + 1e-5)
    # oa = torch.mean((tp + tn) / (tp + fn + fp + tn))
    # oa2 = (tpp + tnn) / (tpp + fnn + fpp + tnn)
    # oa2 = torch.mean(oa2)


    # size = target_oh.size(0) * target_oh.size(1) * target_oh.size(2) * target_oh.size(3)
    # corr = torch.sum(inp == ta)
    # oaa = float(corr) / float(size)
    acc = torch.mean(tp / (tp + fp + smooth))

    # score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    # score = torch.mean(score)
    iou = tp / (fp + fn + tp + smooth)
    miou = torch.mean(iou)
    # return loss.sum(dim=0).mean()
    return miou.item(), acc.item(), oa.item()


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


def fit_train(args, model, image_paths, label_paths,Epoch,optimizer):

    # epoch_step = args.max_epochs
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    image_path = r'E:\data\YAMATO\train'
    label_path = r'E:\data\YAMATO\labelgray'
    train_num = len(os.listdir(image_path))
    val_num = train_num // 5
    epoch_step = train_num // batch_size
    epoch_step_val = val_num // batch_size

    model_train = model.train().cuda()
    # model_train.cuda()
    print('start train')
    train_image_paths, train_label_paths, val_image_paths, val_label_paths = split_train_val(image_paths,
                                                                                             label_paths,
                                                                                             0)
    db_train = TUDataset(train_image_paths, train_label_paths, mode='train')
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    db_train_val = TUDataset(val_image_paths, val_label_paths, mode='train')
    valloader = DataLoader(db_train_val, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    for epoch in range(Epoch):
        total_loss = 0
        total_f_score = 0
        total_miou = 0
        total_oa = 0
        total_acc = 0
        val_loss = 0
        val_f_score = 0
        val_miou = 0
        val_oa = 0
        val_acc = 0
        lr = args.base_lr
        trep = epoch_step - epoch_step_val
        with tqdm(total=trep , desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in enumerate(trainloader):
                if iteration >= epoch_step:
                    break
                imgs, labels = batch
                imgs = imgs.cuda()
                labels = labels.cuda()
                # optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
                optimizer.zero_grad()
                outputs = model_train(imgs)
                loss = CE_Loss(outputs, labels, num_classes=num_classes)
                dice_loss = Dice_loss(outputs, labels)
                loss = loss + dice_loss
                with torch.no_grad():
                    miou, acc, oa = f_score(outputs, labels)

                loss.backward()
                optimizer.step()
                lr = lr * (1.0 - iteration / epoch_step) ** 0.9

                total_loss += loss.item()
                # total_f_score += _f_score.item()
                total_miou += miou
                total_acc += acc
                total_oa += oa

                pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                    # 'f_score': total_f_score / (iteration + 1),
                                    'lr': get_lr(optimizer),
                                    'miou': total_miou / (iteration + 1),
                                    'acc': total_acc / (iteration + 1),
                                    'OA': total_oa / (iteration + 1)})
                pbar.update(1)

        if epoch % 10 == 0:
            model.eval().cuda()
            print('Start Val')
            with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
                for iteration, batch in enumerate(valloader):
                    if iteration >= epoch_step_val:
                        break
                    imgs, labels = batch
                    imgs = imgs.cuda()
                    labels = labels.cuda()
                    with torch.no_grad():
                        outputs = model(imgs)
                        loss = CE_Loss(outputs, labels, num_classes=num_classes)
                        main_dice = Dice_loss(outputs, labels)
                        loss = loss + main_dice
                        miou, acc, oa = f_score(outputs, labels)

                        val_loss += loss.item()
                            # val_f_score += _f_score.item()
                        val_miou += miou
                        val_acc += acc
                        val_oa += oa

                    pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                                    # 'f_score': val_f_score / (iteration + 1),
                                    'lr': get_lr(optimizer),
                                    'mIou': val_miou / (iteration + 1),
                                    'val_acc': val_acc / (iteration + 1),
                                    "val_OA": val_oa / (iteration + 1)})
                    pbar.update(1)
            print('acc: %.3f || Val Loss: %.3f || mIou: %.3f' % (val_acc / (epoch_step_val), val_loss / (epoch_step_val), val_miou / (epoch_step_val)))
            print('savemodel:logs/ep%03d-loss%.3f-val_loss%.3f.pth' % ((epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val))
            torch.save(model.state_dict(), 'savemodel/ep%03d-loss%.3f-val_loss%.3f-acc%.3f.pth' % ((epoch + 1), total_loss / (epoch_step), val_loss / (epoch_step_val), val_acc / (iteration + 1)))
        # lr_scheduler.step()


def trainn(args, model, image_paths, label_paths, epoch, optimizer):
    image_path = r'D:\train'
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    train_num = len(os.listdir(image_path))
    val_num = train_num // 5
    epoch_step = train_num // batch_size
    epoch_step_val = val_num // batch_size
    train_image_paths, train_label_paths, val_image_paths, val_label_paths = split_train_val(image_paths,
                                                                                             label_paths,
                                                                                             0)
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    db_train = TUDataset(train_image_paths, train_label_paths, mode='train')
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    db_train_val = TUDataset(val_image_paths, val_label_paths, mode='val')
    valloader = DataLoader(db_train_val, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    # iterator = tqdm(range(epoch))
    lr = args.base_lr
    max_iterations = epoch * len(trainloader)
    iter_num = 0
    ttacc = 0
    ttmiou = 0
    ttoa = 0
    ttloss = 0
    vvacc = 0
    vvmiou = 0
    vvoa = 0
    vvloss = 0

    model = model.cuda()

    iter_num = 0
    for epoch_num in range(epoch):
        vvacc = 0
        vvmiou = 0
        vvoa = 0
        vvloss = 0
        ttacc = 0
        ttmiou = 0
        ttoa = 0
        ttloss = 0
        model.train()
        trep = epoch_step - epoch_step_val
        with tqdm(total=trep, desc=f'Epoch {epoch_num + 1}/{epoch}', postfix=dict, mininterval=0.3) as pbar:
            for i, sampled in enumerate(trainloader):
                image_batch, label_batch = sampled
                image_batch = image_batch.cuda()
                label_batch = label_batch.cuda()
                outputs = model(image_batch).cuda()

                loss_ce = ce_loss(outputs, label_batch[:].long())

                loss_dice = dice_loss(outputs, label_batch, softmax=True)

                loss = 0.5 * loss_ce + 0.5 * loss_dice

                with torch.no_grad():
                    tmiou, tacc, toa = f_score(outputs, label_batch)

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                ttacc += tacc

                ttmiou += tmiou

                ttoa += toa

                ttloss += loss.item()

                lr_ = lr * (1.0 - iter_num / max_iterations) ** 0.9

                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
                iter_num = iter_num + 1
                pbar.set_postfix(**{'lr': get_lr(optimizer),
                                    'mIou': ttmiou / (i + 1),
                                    'acc': ttacc / (i + 1),
                                    })
                pbar.update(1)
        print('acc: %.3f ||  oa: %.3f || mIou: %.3f  || loss: %.3f' % (ttacc / (i+1), ttoa / (i+1), ttmiou / (i+1), ttloss / (i+1)))
        if (epoch_num+1) % 10 == 0 or epoch_num == 0:
            model.eval()
            with tqdm(total=epoch_step_val, desc=f'Epoch {epoch_num + 1}/{epoch}', postfix=dict, mininterval=0.3) as pbar:
                for j, vsampled in enumerate(valloader):

                    if j >= epoch_step_val:
                        break

                    vimage_batch, vlabel_batch = vsampled
                    vimage_batch = vimage_batch.cuda()
                    vlabel_batch = vlabel_batch.cuda()
                    with torch.no_grad():
                        outputss = model(vimage_batch).cuda()

                        vloss_ce = ce_loss(outputss, vlabel_batch[:].long())

                        vloss_dice = dice_loss(outputss, vlabel_batch, softmax=True)

                        vloss = 0.5 * vloss_ce + 0.5 * vloss_dice

                        vmiou, vacc, voa = f_score(outputss, vlabel_batch)

                        vvmiou += vmiou

                        vvacc += vacc

                        vvoa += voa

                        vvloss += vloss
                    pbar.set_postfix(**{'lr': get_lr(optimizer),
                                        'mIou': vvmiou / (j + 1),
                                        'acc': vvacc / (j + 1),
                                        })
                    pbar.update(1)
            print('valacc: %.3f ||  valoa: %.3f || valmIou: %.3f  || valloss: %.3f' % (vvacc / (j+1), vvoa / (j+1), vvmiou / (j+1), vvloss / (j+1)))
            if epoch_num != 0:
                torch.save(model.state_dict(), 'savemodel/ep%03d-loss%.3f-acc%.3f.pth' % ((epoch_num + 1), vvloss / (epoch_step_val),  vvacc / (epoch_step_val)))




