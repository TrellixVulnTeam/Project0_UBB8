#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
# 将图片编码成rle格式
def rle_encode(img, min_max_threshold=1e-3, max_mean_threshold=None):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    if np.max(img) < min_max_threshold:
        return ''  ## no need to encode if it's all zeros
    if max_mean_threshold and np.mean(img) > max_mean_threshold:
        return ''  ## ignore overfilled mask
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# 将图片从rle解码
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype=np.uint8)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks |= rle_decode(mask)
    return all_masks


# 将目标路径下的rle文件中所包含的所有rle编码，保存到save_img_dir中去
def rle_2_img(train_rle_dir, save_img_dir):
    masks = pd.read_csv(train_rle_dir)
    not_empty = pd.notna(masks.EncodedPixels)
    print(not_empty.sum(), 'masks in', masks[not_empty].ImageId.nunique(), 'images')
    print((~not_empty).sum(), 'empty images in', masks.ImageId.nunique(), 'total images')
    all_batchs = list(masks.groupby('ImageId'))
    train_images = []
    train_masks = []
    i = 0
    for img_id, mask in all_batchs:
        c_mask = masks_as_image(mask['EncodedPixels'].values)
        im = Image.fromarray(c_mask)
        im.save(save_img_dir + img_id.split('.')[0] + '.png')
        print(i, img_id.split('.')[0] + '.png')
        i += 1

    return train_images, train_masks


if __name__ == '__main__':
    rle_2_img('D://data//kaggle-airbus-ship-detection//train_ship_segmentations_v2.csv',
              'D://data//kaggle-airbus-ship-detection//')
