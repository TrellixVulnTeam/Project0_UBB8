import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
import copy
from trainer import TUDataset
from torch.utils.data import DataLoader
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
import argparse
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
# from datasets.dataset_synapse import Synapse_dataset
from trainer import TUDataset
import colorsys

def preprocess_input(image):
    image /= 255.0
    return image


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[-2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def resize_image(image, size):
    iw, ih  = image.size
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh


def detect_image(self, image):
    image = cvtColor(image)

    old_img = copy.deepcopy(image)
    orininal_h = np.array(image).shape[0]
    orininal_w = np.array(image).shape[1]

    image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))

    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

    with torch.no_grad():
        images = torch.from_numpy(image_data)
        if self.cuda:
            images = images.cuda()

        pr = self.net(images)[0]

        pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)

        pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
             int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]

    seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
    for c in range(self.num_classes):
        seg_img[:, :, 0] += ((pr[:, :] == c) * (self.colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c) * (self.colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c) * (self.colors[c][2])).astype('uint8')

    image = Image.fromarray(np.uint8(seg_img)).resize((orininal_w, orininal_h))

    if self.blend:
        image = Image.blend(old_img, image, 0.7)

    return image

image_path = r'E:\data\YAMATO\test\000085.tif'
dataset_config = {
    'Synapse': {
        'Dataset': TUDataset,
        'volume_path': '../data/Synapse/test_vol_h5',
        'list_dir': './lists/lists_Synapse',
        'num_classes': 6,
        'z_spacing': 1,
    },
}
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
# parser.add_argument('--vit_name', type=str, default='ViT-B_16', help='select one vit model')
parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
parser.add_argument('--img_size', type=int, default=512, help='input patch size of network input')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
args = parser.parse_args()

dataset_name = args.dataset
args.num_classes = dataset_config[dataset_name]['num_classes']
args.volume_path = dataset_config[dataset_name]['volume_path']
args.Dataset = dataset_config[dataset_name]['Dataset']
args.list_dir = dataset_config[dataset_name]['list_dir']
args.z_spacing = dataset_config[dataset_name]['z_spacing']
args.is_pretrain = True

config_vit = CONFIGS_ViT_seg[args.vit_name]
config_vit.n_classes = args.num_classes
config_vit.n_skip = args.n_skip
if args.vit_name.find('R50') != -1:
    config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))



class TransU(object):
    _defaults = {
        'model_path': 'savemodel/ep151-loss0.088-acc0.888.pth',
        'num_classes': 6,
        'input_shape': [],
        'blend': True,
        'cuda': True,
        'batch_size': '4',
        'classifier': 'seg'
    }


    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name,value in kwargs.items():
            setattr(self,name,value)
        self.net = ViT_seg(config_vit, img_size=args.img_size, num_classes=args.num_classes).cuda()
        if self.num_classes <= 21:
            self.colors = [(128, 64, 12) , (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
                            (0, 0, 0)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        # self.inference()


    def inference(self,image_path, test_save_path = None):
        # super(TransU).__init__(self)
        model = self.net
        image_path = Image.open(image_path)
        old_img = copy.deepcopy(image_path)
        orininal_h = np.array(image_path).shape[0]
        orininal_w = np.array(image_path).shape[1]
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_path, np.float32)), (2, 0, 1)), 0)
        model.load_state_dict(torch.load(self.model_path))
        # db_train = TUDataset(image_path,  mode='test')
        # testloader = DataLoader(db_train, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        model.eval()
        with torch.no_grad():
            images = torch.from_numpy(image_data).cuda()
            pr = model(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
        seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
        for c in range(self.num_classes):
            seg_img[:, :, 0] += ((pr[:, :] == c) * (self.colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((pr[:, :] == c) * (self.colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((pr[:, :] == c) * (self.colors[c][2])).astype('uint8')
        image = Image.fromarray(np.uint8(seg_img)).resize((orininal_w, orininal_h))
        image = Image.blend(old_img, image, 0.7)
        image.show()
        return image
        # for i,sample_test in enumerate(testloader):
        #     sample_test = sample_test.cuda()


TransU().inference(image_path=r'E:\data\YAMATO\test\001557.tif')


# if __name__ == "__main__":
#     mode = 'predict'
#     if mode == "predict":
#         while True:
#             img = input('input image filename:')
#             try:
#                 image = Image.open(img)
#             except:
#                 print('ERROR')
#             else:
#                 r_image = model().detect_image(image)
#                 r_image.show()


