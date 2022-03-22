import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainn
import glob
from torchsummary import summary
import torch.optim as optim
from UNET import UNet
from resunet import *
from doubleunet import DUNet
from doubleunet_2 import DoubleUnet
from nets import att_deeplabv3_plus, deeplabv3_plus
from nets import trans_deeplab
import ml_collections
from nets import pooltran_deeplab
import math
import model_v3
import biesenetv2
import efficientnetv2, VANFPN, OCR, VANOCR, HROCR
from maskf1 import efficientnetv2_s


parser = argparse.ArgumentParser()

parser.add_argument('--num_classes', type=int,
                    # default=9, help='output channel of network')
                    default=6, help='output channel of network')
parser.add_argument('--max_epochs', type=int,
                    default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=4, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=0,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    # default=224, help='input patch size of network input')
                    default=512, help='input patch size of network input')
args = parser.parse_args()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    # if not os.path.exists(snapshot_path):
    # os.makedirs(snapshot_path)
    #
    # config_vit = CONFIGS_ViT_seg[args.vit_name]
    # config_vit.n_classes = args.num_classes
    # config_vit.n_skip = args.n_skip
    # if args.vit_name.find('R50') != -1:
    #     config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    # net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    # net = efficientnetv2_s(6).cuda()

    # summary(net, (3, 512, 512))
    # net.load_from(weights=np.load(config_vit.pretrained_path))

    # net.load_state_dict(torch.load('savemodel/ep151-loss0.088-acc0.888.pth'))

    # net = UNet(3, 6).cuda()
    # net = DUNet().cuda()
    # import torchvision.models as models
    # base_model = models.vgg19_bn()
    # net = DoubleUnet(base_model).cuda()
    # net = resnet101(3, 6, pretrain=True).cuda()

    config_deep = ml_collections.ConfigDict()
    config_deep.transformer = ml_collections.ConfigDict()
    config_deep.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config_deep.hidden_size = 768
    config_deep.transformer.num_heads = 12
    config_deep.transformer.num_layers = 6
    config_deep.transformer.mlp_dim = 3072
    config_deep.transformer.attention_dropout_rate = 0.0
    config_deep.transformer.dropout_rate = 0.1

    # net = trans_deeplab.DeepLab(num_classes=6, backbone="mobilenet", config=config_deep, downsample_factor=16, pretrained=False).cuda()
    # net = deeplabv3_plus.DeepLab(num_classes=6, backbone="xception", pretrained=True, downsample_factor=16).cuda()
    # net = pooltran_deeplab.poolformer_s12(pretrained=False)
    # net = model_v3.mobilenet_v3_small().cuda()
    # net = biesenetv2.BiSeNetV2(n_classes=6).cuda()
    # net = efficientnetv2.efficientnetv2_s().cuda()

    # net = VAN.van_tiny().cuda()
    net = VANOCR.van_tiny().cuda()
    # net = OCR.HighResolutionNet().cuda()
    # net = HROCR.HighResolutionNet().cuda()
    # summary(net, (3, 512, 512))

    # xx = torch.randn(2,3,512,512).cuda()
    # yy = net(xx)
    # print(yy.size())

    image_paths = glob.glob(r'D:\train\*tif')
    image_paths.sort()
    label_paths = glob.glob(r'D:\labelgray\*tif')
    label_paths.sort()
    epoch = args.max_epochs
    # trainer[dataset_name](args, net, snapshot_path, trainpath, trainlabel )
    lr = 2e-4  #2E-4

    # optimizer = optim.Adam(net.parameters(), lr, weight_decay=5e-4)
    # lr = args.base_lr

    # optimizer = optim.SGD(net.parameters(), lr=2e-2, weight_decay=0.0001)
    # optimizer = optim.Adam(net.parameters(), lr, weight_decay=5e-4)
    optimizer = torch.optim.AdamW(net.parameters(),
                                  lr=lr,
                                  betas=(0.9, 0.999),
                                  # weight_decay=0.01,
                                  weight_decay=0.0001,
                                  )
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)
    from keras.optimizer_v2 import learning_rate_schedule as klropt

    # lr_scheduler =klropt.PolynomialDecay(initial_learning_rate=lr, power=0.9,)
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

    image_path = r'D:\train'
    num_classes = args.num_classes
    batch_size = args.batch_size
    train_num = len(os.listdir(image_path))
    val_num = math.ceil(len(os.listdir(image_path)) // 5)
    epoch_step = train_num // batch_size
    epoch_step_val = val_num // batch_size
    train_image_paths, train_label_paths, val_image_paths, val_label_paths = split_train_val(image_paths,
                                                                                             label_paths,
                                                                                             0)

    trainn(num_classes, net, train_image_paths, val_image_paths, epoch_step, epoch_step_val, train_label_paths,
           val_label_paths, epoch, optimizer, batch_size)


