import numpy as np
import os
import rawpy
import torch
from torch.utils import data

import sys
sys.path.append('/root/autodl-tmp/RetinexRawMamba')
from utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class SIDSonyDataset(data.Dataset):
    def __init__(self, data_dir, image_list_file, patch_size=None, split='train', transpose=False,
                h_flip=False, v_flip=False, ratio=True, **kwargs):
        assert os.path.exists(data_dir), "data_dir: {} not found.".format(data_dir)
        self.data_dir = data_dir
        image_list_file = os.path.join(data_dir, image_list_file)
        assert os.path.exists(image_list_file), "image_list_file: {} not found.".format(image_list_file)
        self.image_list_file = image_list_file
        self.patch_size = patch_size
        self.split = split
        self.transpose = transpose
        self.h_flip = h_flip
        self.v_flip = v_flip
        self.ratio = ratio
        self.black_level = 512
        self.white_level = 16383
        
        self.img_info = []
        with open(self.image_list_file, 'r') as f:
            for i, img_pair in enumerate(f):
                img_pair = img_pair.strip()  # ./Sony/short/10003_00_0.04s.ARW ./Sony/long/10003_00_10s.ARW
                input_path, gt_path, iso, focus = img_pair.split(' ')
                input_exposure = float(os.path.split(input_path)[-1][9:-5]) # 0.04
                gt_exposure = float(os.path.split(gt_path)[-1][9:-5]) # 10
                ratio = min(gt_exposure/input_exposure, 300)
                input_exposure = float(os.path.split(input_path)[-1][9:-5]) # 0.04
                gt_exposure = float(os.path.split(gt_path)[-1][9:-5]) # 10
                ratio = min(gt_exposure/input_exposure, 300)
                self.img_info.append({
                    'input_path': input_path,
                    'gt_path': gt_path,
                    'input_exposure': input_exposure,
                    'gt_exposure': gt_exposure,
                    'ratio': np.float32(ratio),
                    'iso': float(iso[3::]),
                    'focus': focus,
                })
        print("processing: {} images for {}".format(len(self.img_info), self.split))

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, index):
        info = self.img_info[index]
        input_path = info['input_path']#./Sony/short/10003_00_0.04s.ARW 
        input_raw = rawpy.imread(os.path.join(self.data_dir, input_path))
        input_raw = self.pack_raw(input_raw)#(4,2848/2,4256/2) numpy

        gt_path = info['gt_path']
        gt_raw = rawpy.imread(os.path.join(self.data_dir, gt_path))
        gt_rgb = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        gt_rgb = gt_rgb.transpose(2, 0, 1)#未归一 (3,2848,4256) numpy
        gt_raw = self.pack_raw(gt_raw)#(4,2848/2,4256/2) numpy

        input_raw = (np.float32(input_raw) - self.black_level) / np.float32(self.white_level - self.black_level)
        gt_raw = (np.float32(gt_raw) - self.black_level) / np.float32(self.white_level - self.black_level)
        gt_rgb = np.float32(gt_rgb) / np.float32(65535)
        if self.ratio:
            input_raw = input_raw * info['ratio']
        input_raw = np.maximum(np.minimum(input_raw, 1.0), 0.0)
        gt_raw = np.maximum(np.minimum(gt_raw, 1.0), 0.0)

        if self.split == 'train':
            if self.h_flip and np.random.randint(0,2) == 1:  # random horizontal flip
                input_raw = np.flip(input_raw, axis=2)
                gt_raw = np.flip(gt_raw, axis=2)
                gt_rgb = np.flip(gt_rgb, axis=2)
            if self.v_flip and np.random.randint(0,2) == 1:  # random vertical flip
                input_raw = np.flip(input_raw, axis=1)
                gt_raw = np.flip(gt_raw, axis=1)
                gt_rgb = np.flip(gt_rgb, axis=1)
            if self.transpose and np.random.randint(0,2) == 1:  # random transpose
                input_raw = np.transpose(input_raw, (0, 2, 1))
                gt_raw = np.transpose(gt_raw, (0, 2, 1)) 
                gt_rgb = np.transpose(gt_rgb, (0, 2, 1)) 
            if self.patch_size:
                input_patch, gt_raw_patch, gt_rgb_patch = self.crop_random_patch(input_raw, gt_raw, gt_rgb, self.patch_size)
                input_raw = input_patch.copy()
                gt_raw = gt_raw_patch.copy()
                gt_rgb = gt_rgb_patch.copy()
        
        input_raw = np.ascontiguousarray(input_raw)
        gt_raw = np.ascontiguousarray(gt_raw)
        gt_rgb = np.ascontiguousarray(gt_rgb)

        input_raw = torch.from_numpy(input_raw).float()
        gt_raw = torch.from_numpy(gt_raw).float()
        gt_rgb = torch.from_numpy(gt_rgb).float()

        return {
            'input_raw': input_raw,
            'gt_raw': gt_raw,
            'gt_rgb': gt_rgb,
            'input_path': input_path,
            'gt_path': gt_path,
            'input_exposure': info['input_exposure'],
            'gt_exposure': info['gt_exposure'],
            'ratio': info['ratio']
        }

    def pack_raw(self, raw):
        # pack Bayer image to 4 channels (RGBG)
        im = raw.raw_image_visible.astype(np.uint16)
        H, W = im.shape
        im = np.expand_dims(im, axis=0)
        out = np.concatenate((im[:, 0:H:2, 0:W:2],
                            im[:, 0:H:2, 1:W:2],
                            im[:, 1:H:2, 1:W:2],
                            im[:, 1:H:2, 0:W:2]), axis=0)
        return out

    def crop_random_patch(self, input_raw, gt_raw, gt_rgb, patch_size):
        '''
        input_raw, gt_raw: numpy with shape (4,H/2,W/2)
        gt_rgb: numpy with shape (3,H,W)
        '''
        _, H, W = input_raw.shape
        yy, xx = np.random.randint(0, H - patch_size),  np.random.randint(0, W - patch_size)
        input_raw = input_raw[:, yy:yy + patch_size, xx:xx + patch_size]
        gt_raw = gt_raw[:, yy:yy + patch_size, xx:xx + patch_size]
        gt_rgb = gt_rgb[:, yy*2:(yy + patch_size)*2 , xx*2:(xx + patch_size)*2]
        return input_raw, gt_raw, gt_rgb


@DATASET_REGISTRY.register()
class SIDFujiDataset(data.Dataset):
    def __init__(self, data_dir, image_list_file, patch_size=None, split='train', transpose=False,
                h_flip=False, v_flip=False, ratio=True, **kwargs):
        assert os.path.exists(data_dir), "data_dir: {} not found.".format(data_dir)
        self.data_dir = data_dir
        image_list_file = os.path.join(data_dir, image_list_file)
        assert os.path.exists(image_list_file), "image_list_file: {} not found.".format(image_list_file)
        self.image_list_file = image_list_file
        self.patch_size = patch_size
        self.split = split
        self.transpose = transpose
        self.h_flip = h_flip
        self.v_flip = v_flip
        self.ratio = ratio
        self.black_level = 1024
        self.white_level = 16383

        self.img_info = []
        with open(self.image_list_file, 'r') as f:
            for i, img_pair in enumerate(f):
                img_pair = img_pair.strip()  # ./Sony/short/10003_00_0.04s.ARW ./Sony/long/10003_00_10s.ARW ISO200 F9
                input_path, gt_path, iso, focus = img_pair.split(' ')
                input_exposure = float(os.path.split(input_path)[-1][9:-5]) # 0.04
                gt_exposure = float(os.path.split(gt_path)[-1][9:-5]) # 10
                ratio = min(gt_exposure/input_exposure, 300)
                self.img_info.append({
                    'input_path': input_path,
                    'gt_path': gt_path,
                    'input_exposure': input_exposure,
                    'gt_exposure': gt_exposure,
                    'ratio': np.float32(ratio),
                    'iso': float(iso[3::]),
                    'focus': focus,
                })
        print("processing: {} images for {}".format(len(self.img_info), self.split))

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, index):
        info = self.img_info[index]
        input_path = info['input_path']#./Sony/short/10003_00_0.04s.ARW 
        input_raw = rawpy.imread(os.path.join(self.data_dir, input_path))
        input_raw = self.pack_raw(input_raw)#(4,2848/2,4256/2) numpy

        gt_path = info['gt_path']
        gt_raw = rawpy.imread(os.path.join(self.data_dir, gt_path))
        gt_rgb = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        gt_rgb = gt_rgb.transpose(2, 0, 1)#未归一 (3,2848,4256) numpy
        gt_raw = self.pack_raw(gt_raw)#(4,2848/2,4256/2) numpy
        
        packed_H, packed_W = gt_raw.shape[1:]
        #to ensure the shape of the output of the model is same as gt_rgb
        gt_rgb = gt_rgb[:, :3*packed_H, :3*packed_W]

        input_raw = (np.float32(input_raw) - self.black_level) / np.float32(self.white_level - self.black_level)
        gt_raw = (np.float32(gt_raw) - self.black_level) / np.float32(self.white_level - self.black_level)
        gt_rgb = np.float32(gt_rgb) / np.float32(65535)
        
        if self.ratio:
            input_raw = input_raw * info['ratio']
        input_raw = np.maximum(np.minimum(input_raw, 1.0), 0.0)
        gt_raw = np.maximum(np.minimum(gt_raw, 1.0), 0.0)

        if self.split == 'train':
            if self.h_flip and np.random.randint(0,2) == 1:  # random horizontal flip
                input_raw = np.flip(input_raw, axis=2)
                gt_raw = np.flip(gt_raw, axis=2)
                gt_rgb = np.flip(gt_rgb, axis=2)
            if self.v_flip and np.random.randint(0,2) == 1:  # random vertical flip
                input_raw = np.flip(input_raw, axis=1)
                gt_raw = np.flip(gt_raw, axis=1)
                gt_rgb = np.flip(gt_rgb, axis=1)
            if self.transpose and np.random.randint(0,2) == 1:  # random transpose
                input_raw = np.transpose(input_raw, (0, 2, 1))
                gt_raw = np.transpose(gt_raw, (0, 2, 1)) 
                gt_rgb = np.transpose(gt_rgb, (0, 2, 1)) 
            if self.patch_size:
                input_patch, gt_raw_patch, gt_rgb_patch = self.crop_random_patch(input_raw, gt_raw, gt_rgb, self.patch_size)
                input_raw = input_patch.copy()
                gt_raw = gt_raw_patch.copy()
                gt_rgb = gt_rgb_patch.copy()
        
        input_raw = np.ascontiguousarray(input_raw)
        gt_raw = np.ascontiguousarray(gt_raw)
        gt_rgb = np.ascontiguousarray(gt_rgb)

        input_raw = torch.from_numpy(input_raw).float()
        gt_raw = torch.from_numpy(gt_raw).float()
        gt_rgb = torch.from_numpy(gt_rgb).float()

        return {
            'input_raw': input_raw,
            'gt_raw': gt_raw,
            'gt_rgb': gt_rgb,
            'input_path': input_path,
            'gt_path': gt_path,
            'input_exposure': info['input_exposure'],
            'gt_exposure': info['gt_exposure'],
            'ratio': info['ratio']
        }
    
    def pack_raw(self, raw):
        # pack XTrans image to 9 channels ()
        im = raw.raw_image_visible.astype(np.uint16)

        H, W = im.shape
        h1 = 0
        h2 = H // 6 * 6
        w1 = 0
        w2 = W // 6 * 6
        out = np.zeros((9, h2 // 3, w2 // 3), dtype=np.uint16)
        
        # 0 R
        out[0, 0::2, 0::2] = im[h1:h2:6, w1:w2:6]
        out[0, 0::2, 1::2] = im[h1:h2:6, w1+4:w2:6]
        out[0, 1::2, 0::2] = im[h1+3:h2:6, w1+1:w2:6]
        out[0, 1::2, 1::2] = im[h1+3:h2:6, w1+3:w2:6]

        # 1 G
        out[1, 0::2, 0::2] = im[h1:h2:6, w1+2:w2:6]
        out[1, 0::2, 1::2] = im[h1:h2:6, w1+5:w2:6]
        out[1, 1::2, 0::2] = im[h1+3:h2:6, w1+2:w2:6]
        out[1, 1::2, 1::2] = im[h1+3:h2:6, w1+5:w2:6]

        # 1 B
        out[2, 0::2, 0::2] = im[h1:h2:6, w1+1:w2:6]
        out[2, 0::2, 1::2] = im[h1:h2:6, w1+3:w2:6]
        out[2, 1::2, 0::2] = im[h1+3:h2:6, w1:w2:6]
        out[2, 1::2, 1::2] = im[h1+3:h2:6, w1+4:w2:6]

        # 4 R
        out[3, 0::2, 0::2] = im[h1+1:h2:6, w1+2:w2:6]
        out[3, 0::2, 1::2] = im[h1+2:h2:6, w1+5:w2:6]
        out[3, 1::2, 0::2] = im[h1+5:h2:6, w1+2:w2:6]
        out[3, 1::2, 1::2] = im[h1+4:h2:6, w1+5:w2:6]

        # 5 B
        out[4, 0::2, 0::2] = im[h1+2:h2:6, w1+2:w2:6]
        out[4, 0::2, 1::2] = im[h1+1:h2:6, w1+5:w2:6]
        out[4, 1::2, 0::2] = im[h1+4:h2:6, w1+2:w2:6]
        out[4, 1::2, 1::2] = im[h1+5:h2:6, w1+5:w2:6]

        out[5, :, :] = im[h1+1:h2:3, w1:w2:3]
        out[6, :, :] = im[h1+1:h2:3, w1+1:w2:3]
        out[7, :, :] = im[h1+2:h2:3, w1:w2:3]
        out[8, :, :] = im[h1+2:h2:3, w1+1:w2:3]
        return out

    def crop_random_patch(self, input_raw, gt_raw, gt_rgb, patch_size):
        '''
        input_raw, gt_raw: numpy with shape (9,1344,2010)
        gt_rgb: numpy with shape (3,4032,6030)
        '''
        _, H, W = input_raw.shape
        yy, xx = np.random.randint(0, H - patch_size),  np.random.randint(0, W - patch_size)
        input_raw = input_raw[:, yy:yy + patch_size, xx:xx + patch_size]
        gt_raw = gt_raw[:, yy:yy + patch_size, xx:xx + patch_size]
        gt_rgb = gt_rgb[:, yy*3:(yy + patch_size)*3 , xx*3:(xx + patch_size)*3]

        return input_raw, gt_raw, gt_rgb


if __name__=='__main__':
    seed = 3407
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    dataset = SIDSonyDataset(data_dir='/root/autodl-tmp/datasets/SID/Sony',image_list_file='Sony_train_list.txt',
                             split='train',patch_size=512)
    #dataset = SIDFujiDataset(data_dir='/root/autodl-tmp/datasets/SID/Fuji',image_list_file='Fuji_train_list.txt',
    #                        split='train',patch_size=512)
    data = dataset[7]
    input_raw, gt_raw, gt_rgb = data['input_raw'], data['gt_raw'], data['gt_rgb']
    print(input_raw.shape, gt_raw.shape, gt_rgb.shape)
    print(input_raw.min(),input_raw.max())
    print(gt_raw.min(),gt_raw.max())
    print(gt_rgb.min(),gt_rgb.max())

