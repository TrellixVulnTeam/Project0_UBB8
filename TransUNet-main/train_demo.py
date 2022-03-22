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
from nets import att_deeplabv3_plus
from nets import trans_deeplab
import ml_collections


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    # default=9, help='output channel of network')
                    default=6, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=2, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=0,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    # default=224, help='input patch size of network input')
                    default=512, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # dataset_name = args.dataset
    # dataset_config = {
    #     'Synapse': {
    #         'root_path': '../data/Synapse/train_npz',
    #         'list_dir': './lists/lists_Synapse',
    #         'num_classes': 6,
    #     },
    # }
    # args.num_classes = dataset_config[dataset_name]['num_classes']
    # args.root_path = dataset_config[dataset_name]['root_path']
    # args.list_dir = dataset_config[dataset_name]['list_dir']
    # args.is_pretrain = True
    # args.exp = 'TU_' + dataset_name + str(args.img_size)
    # snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    # snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    # snapshot_path += '_' + args.vit_name
    # snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    # snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    # snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    # snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    # snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    # snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    # snapshot_path = snapshot_path + '_'+str(args.img_size)
    # snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    # if not os.path.exists(snapshot_path):
    #     os.makedirs(snapshot_path)
    # config_vit = CONFIGS_ViT_seg[args.vit_name]
    # config_vit.n_classes = args.num_classes
    # config_vit.n_skip = args.n_skip
    # if args.vit_name.find('R50') != -1:
    #     config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    # net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()



    # summary(net, (3, 512, 512))
    # net.load_from(weights=np.load(config_vit.pretrained_path))

    # net.load_state_dict(torch.load('savemodel/ep151-loss0.088-acc0.888.pth'))
    # trainer = {'Synapse': trainer_synapse,}

    # net = UNet(3, 6).cuda()
    # net = DUNet().cuda()
    # import torchvision.models as models
    # base_model = models.vgg19_bn()
    # net = DoubleUnet(base_model).cuda()
    # net = resnet34(3, 6,pretrain=False).cuda()

    config_deep = ml_collections.ConfigDict()
    config_deep.transformer = ml_collections.ConfigDict()
    config_deep.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config_deep.hidden_size = 768
    config_deep.transformer.num_heads = 12
    config_deep.transformer.num_layers = 6
    config_deep.transformer.mlp_dim = 3072
    config_deep.transformer.attention_dropout_rate = 0.0
    config_deep.transformer.dropout_rate = 0.1

    net = trans_deeplab.DeepLab(num_classes=6, backbone='mobilenet', config=config_deep, downsample_factor=16, pretrained=False).cuda()
    summary(net, (3, 512, 512))
    xx = torch.randn(2,3,512,512).cuda()
    yy = net(xx)
    print(yy.size())

    trainpath = glob.glob(r'D:\train\*tif')
    trainlabel =glob.glob(r'D:\labelgray\*tif')
    epoch = args.max_epochs
    # trainer[dataset_name](args, net, snapshot_path, trainpath, trainlabel )
    # lr = args.base_lr
    lr = 0.006
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    # optimizer = optim.Adam(net.parameters(), lr, weight_decay=5e-4)
    # optimizer = torch.optim.AdamW([{'params':filter(lambda p: p.requires_grad, net.parameters()), 'lr': lr}],
    #                               lr=lr,
    #                               betas=(0.9, 0.999),
    #                               weight_decay=0.01,
    #                               )
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)
    trainn(args, net, trainpath, trainlabel, epoch, optimizer,lr)

