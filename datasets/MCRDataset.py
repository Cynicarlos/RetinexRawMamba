import imageio
import numpy as np
import os
import torch

from utils.registry import DATASET_REGISTRY
from torch.utils import data


@DATASET_REGISTRY.register()
class MCRDataset(data.Dataset):

    def __init__(self, data_dir, image_list_file, patch_size=None, split='train',
                 transpose=True, h_flip=True, v_flip=True, ratio=True, **kwargs):

        assert os.path.exists(data_dir), "data_dir: {} not found.".format(data_dir)
        self.data_dir = data_dir #E:\Deep Learning\datasets\MCR
        image_list_file = os.path.join(data_dir, image_list_file)

        assert os.path.exists(image_list_file), "image_list_file: {} not found.".format(image_list_file)
        self.image_list_file = image_list_file#..../MCR_train_list.txt

        self.split = split  
        self.patch_size = patch_size
        self.transpose = transpose
        self.h_flip = h_flip
        self.v_flip = v_flip
        self.ratio = ratio

        self.black_level = 0
        self.white_level = 255

        self.img_info = []

        with open(self.image_list_file, 'r') as f:
            for i, img_pair in enumerate(f):
                input_raw_path, gt_raw_path, gt_rgb_path = img_pair.strip().split(' ')

                img_num = int(input_raw_path[-23:-20])
                input_exposure = int(input_raw_path[-8:-4],16)
                if img_num < 500:
                    gt_exposure = 12287
                else:
                    gt_exposure = 1023
                ratio = gt_exposure / input_exposure

                self.img_info.append({
                    'input_path': input_raw_path,
                    'gt_raw_path': gt_raw_path,
                    'gt_rgb_path': gt_rgb_path,
                    'input_exposure': input_exposure,
                    'gt_exposure': gt_exposure,
                    'ratio': np.float32(ratio)
                })

        print("processing: {} images for {}".format(len(self.img_info), self.split))


    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, index):
        info = self.img_info[index]
        input_path = info['input_path']#./Mono_Colored_RAW_Paired_DATASET/Color_RAW_Input/C00117_48mp_0x8_0x01ff.tif
        gt_raw_path = info['gt_raw_path']
        gt_rgb_path = info['gt_rgb_path']#./Mono_Colored_RAW_Paired_DATASET/RGB_GT/C00117_48mp_0x8_0x2fff.jpg

        input_raw = imageio.imread(os.path.join(self.data_dir,input_path)).transpose(2,0,1)#(1,h,w) numpy
        gt_raw = imageio.imread(os.path.join(self.data_dir,gt_raw_path)).transpose(2,0,1)#(1,h,w) numpy
        gt_rgb = imageio.imread(os.path.join(self.data_dir,gt_rgb_path)).transpose(2,0,1)#(3,1024,1280) numpy

        input_raw = self.pack_raw(input_raw) #(4, h/2, w/2)
        gt_raw = self.pack_raw(gt_raw) #(4, h/2, w/2)

        if self.split == 'train':
            #data argmentation
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
            input_raw, gt_raw, gt_rgb = self.crop_random_patch(input_raw, gt_raw, gt_rgb, self.patch_size)


        input_raw = (np.float32(input_raw) - self.black_level) / np.float32(self.white_level - self.black_level)  # subtract the black level
        gt_raw = (np.float32(gt_raw) - self.black_level) / np.float32(self.white_level - self.black_level)
        gt_rgb = np.float32(gt_rgb) / np.float32(255)

        if self.ratio:
            input_raw = input_raw * info['ratio']
        input_raw = np.maximum(np.minimum(input_raw, 1.0), 0.0)

        gt_rgb = gt_rgb.clip(0.0, 1.0)

        input_raw = torch.from_numpy(input_raw).float()
        gt_raw = torch.from_numpy(gt_raw).float()
        gt_rgb = torch.from_numpy(gt_rgb).float()
        

        return {
            'input_raw': input_raw,
            'gt_raw': gt_raw,
            'gt_rgb': gt_rgb,
            'input_path': input_path,
            'gt_path': gt_rgb_path,
            'input_exposure': info['input_exposure'],
            'gt_exposure': info['gt_exposure'],
            'ratio': info['ratio']
        }

    def pack_raw(self, image):
        _, H, W = image.shape
        out = np.concatenate((image[:, 0:H:2, 0:W:2],
                            image[:, 0:H:2, 1:W:2],
                            image[:, 1:H:2, 1:W:2],
                            image[:, 1:H:2, 0:W:2]), axis=0)
        return out

    def crop_random_patch(self, input_raw, gt_raw, gt_rgb, patch_size):
        '''
        input_raw, gt_raw: numpy with shape (4,512,640)
        gt_rgb: numpy with shape (3,1024,1280)
        '''
        _, H, W = input_raw.shape
        yy, xx = np.random.randint(0, H - patch_size + 1),  np.random.randint(0, W - patch_size + 1)
        input_raw = input_raw[:, yy:yy + patch_size, xx:xx + patch_size]
        gt_raw = gt_raw[:, yy:yy + patch_size, xx:xx + patch_size]
        gt_rgb = gt_rgb[:, yy*2:(yy + patch_size)*2 , xx*2:(xx + patch_size)*2]

        return input_raw, gt_raw, gt_rgb

if __name__=='__main__':
    dataset = MCRDataset(data_dir='E:/Deep Learning/datasets/MCR',image_list_file='MCR_test_list.txt', data_type='test',patch_size=None)
    data = dataset[7]
    input_raw = data['input_raw']
    gt_raw = data['gt_raw']
    gt_rgb = data['gt_rgb']
    gt_raw_A = data['gt_raw_amplitute']
    gt_raw_P = data['gt_raw_phase']
    print(input_raw.shape, gt_raw.shape, gt_rgb.shape, gt_raw_A.shape, gt_raw_P.shape)
    print(input_raw.min(),input_raw.max())
    print(gt_raw.min(),gt_raw.max())
    print(gt_rgb.min(),gt_rgb.max())
    print(gt_raw_A.min(), gt_raw_A.max())
    print(gt_raw_P.min(), gt_raw_P.max())
