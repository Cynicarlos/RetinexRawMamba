import numpy as np
import os
import rawpy
import torch
from torch.utils import data

import sys
sys.path.append('E:\Deep Learning\IATlab\TBDNet')
from utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class SIDSonyDataset(data.Dataset):
    def __init__(self, data_dir, image_list_file, patch_size=None, data_type='train', 
                max_clip=1.0, min_clip=None, transpose=False,
                h_flip=False, v_flip=False, ratio=True, merge_test=False, **kwargs):
        """
        :param data_path: dataset directory
        :param image_list_file: contains image file names under data_path
        :param patch_size: if None, full images are returned, otherwise patches are returned
        :param data_type: train or valid

        """
        assert os.path.exists(data_dir), "data_path: {} not found.".format(data_dir)
        self.data_path = data_dir
        image_list_file = os.path.join(data_dir, image_list_file)
        assert os.path.exists(image_list_file), "image_list_file: {} not found.".format(image_list_file)
        self.image_list_file = image_list_file#..../Sony_train_list.txt
        self.patch_size = patch_size
        self.data_type = data_type
        self.max_clip = max_clip
        self.min_clip = min_clip
        self.transpose = transpose
        self.h_flip = h_flip
        self.v_flip = v_flip
        self.ratio = ratio
        self.merge_test = merge_test
        #self.only_00 = only_00
        self.black_level = 512
        self.white_level = 16383

        self.img_info = []
        with open(self.image_list_file, 'r') as f:
            for i, img_pair in enumerate(f):
                img_pair = img_pair.strip()  # ./Sony/short/10003_00_0.04s.ARW ./Sony/long/10003_00_10s.ARW ISO200 F9
                input_path, gt_path, iso, focus = img_pair.split(' ')
                #if self.data_type == 'test' and self.only_00:
                #    if os.path.split(input_path)[-1][5:8] != '_00':
                #        continue
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
        print("processing: {} images for {}".format(len(self.img_info), self.data_type))

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, index):
        info = self.img_info[index]
        input_path = info['input_path']#./Sony/short/10003_00_0.04s.ARW 
        input_raw = rawpy.imread(os.path.join(self.data_path, input_path))
        input_raw = self.pack_raw(input_raw)#(4,2848/2,4256/2) numpy

        gt_path = info['gt_path']
        gt_raw = rawpy.imread(os.path.join(self.data_path, gt_path))
        gt_rgb = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        gt_rgb = gt_rgb.transpose(2, 0, 1)#未归一 (3,2848,4256) numpy

        gt_raw = self.pack_raw(gt_raw)#(4,2848/2,4256/2) numpy

        if self.data_type == 'test' and self.merge_test:
            input_raw = self.crop_vertical_patch(input_raw)#list

            input_raw = [
                (np.float32(t) - self.black_level) / np.float32(self.white_level - self.black_level)
                for t in input_raw
            ]

            gt_raw = (np.float32(gt_raw) - self.black_level) / np.float32(self.white_level - self.black_level)
            gt_rgb = np.float32(gt_rgb) / np.float32(65535)

            if self.ratio:
                input_raw = [t * info['ratio'] for t in input_raw]
            if self.max_clip is not None:
                input_raw = [np.minimum(t, self.max_clip) for t in input_raw]
            if self.min_clip is not None:
                input_raw = [np.maximum(t, self.min_clip) for t in input_raw]

            gt_rgb = gt_rgb.clip(0.0, 1.0)

            input_raw = [torch.from_numpy(t).float() for t in input_raw]
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
        
        else:
            if self.patch_size:
                input_patch, gt_raw_patch, gt_rgb_patch = self.crop_random_patch(input_raw, gt_raw, gt_rgb, self.patch_size)
                if self.h_flip and np.random.randint(0,2) == 1 and self.data_type == 'train':  # random horizontal flip
                    input_patch = np.flip(input_patch, axis=2)
                    gt_rgb_patch = np.flip(gt_rgb_patch, axis=2)
                    gt_raw_patch = np.flip(gt_raw_patch, axis=2)
                if self.v_flip and np.random.randint(0,2) == 1 and self.data_type == 'train':  # random vertical flip
                    input_patch = np.flip(input_patch, axis=1)
                    gt_rgb_patch = np.flip(gt_rgb_patch, axis=1)
                    gt_raw_patch = np.flip(gt_raw_patch, axis=1)
                if self.transpose and np.random.randint(0,2) == 1 and self.data_type == 'train':  # random transpose
                    input_patch = np.transpose(input_patch, (0, 2, 1))
                    gt_rgb_patch = np.transpose(gt_rgb_patch, (0, 2, 1))
                    gt_raw_patch = np.transpose(gt_raw_patch, (0, 2, 1)) 

                input_raw = input_patch.copy()
                gt_raw = gt_raw_patch.copy()
                gt_rgb = gt_rgb_patch.copy()
            

            input_raw = (np.float32(input_raw) - self.black_level) / np.float32(self.white_level - self.black_level)
            gt_raw = (np.float32(gt_raw) - self.black_level) / np.float32(self.white_level - self.black_level)
            gt_rgb = np.float32(gt_rgb) / np.float32(65535)

            if self.ratio:
                input_raw = input_raw * info['ratio']
            if self.max_clip is not None:
                input_raw = np.minimum(input_raw, self.max_clip)
            if self.min_clip is not None:
                input_raw = np.maximum(input_raw, self.min_clip)

            gt_rgb = gt_rgb.clip(0.0, 1.0)

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


    def crop_vertical_patch(self, input_raw):
        '''
        input_raw, gt_raw: numpy with shape (4,H,W)
        '''
        _, H, W = input_raw.shape
        input_raws = [input_raw[:,:,0:W//2],input_raw[:,:,W//2:]]
        return input_raws

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

    def crop_center_patch(self, input_raw, gt_raw, gt_rgb, patch_size):
        '''
        input_raw, gt_raw: numpy with shape (4,H/2,W/2)
        gt_rgb: numpy with shape (3,H,W)
        '''
        _, H, W = input_raw.shape
        yy, xx = (H - patch_size) // 2,  (W - patch_size) // 2
        input_raw = input_raw[:, yy:yy + patch_size, xx:xx + patch_size]
        gt_raw = gt_raw[:, yy:yy + patch_size, xx:xx + patch_size]
        gt_rgb = gt_rgb[:, yy*2:(yy + patch_size)*2 , xx*2:(xx + patch_size)*2]
    
        return input_raw, gt_raw, gt_rgb


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

@DATASET_REGISTRY.register()
class SIDFujiDataset(data.Dataset):
    def __init__(self, data_dir, image_list_file, patch_size=None, data_type='train',
                max_clip=1.0, min_clip=0.0,transpose=False, h_flip=False, v_flip=False,
                ratio=True, merge_test=False, **kwargs):
        assert os.path.exists(data_dir), "data_path: {} not found.".format(data_dir)
        self.data_path = data_dir
        image_list_file = os.path.join(data_dir, image_list_file)
        assert os.path.exists(image_list_file), "image_list_file: {} not found.".format(image_list_file)
        self.image_list_file = image_list_file#..../Sony_train_list.txt
        self.patch_size = patch_size
        self.data_type = data_type
        self.max_clip = max_clip
        self.min_clip = min_clip
        self.transpose = transpose
        self.h_flip = h_flip
        self.v_flip = v_flip
        self.ratio = ratio
        self.merge_test = merge_test
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
        print("processing: {} images for {}".format(len(self.img_info), self.data_type))

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, index):
        info = self.img_info[index]
        input_path = info['input_path']#./Sony/short/10003_00_0.04s.ARW 
        input_raw = rawpy.imread(os.path.join(self.data_path, input_path))
        input_raw = self.pack_raw(input_raw)#(4,2848/2,4256/2) numpy

        gt_path = info['gt_path']
        gt_raw = rawpy.imread(os.path.join(self.data_path, gt_path))
        gt_rgb = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        gt_rgb = gt_rgb.transpose(2, 0, 1)#未归一 (3,2848,4256) numpy
        gt_raw = self.pack_raw(gt_raw)#(4,2848/2,4256/2) numpy
        packed_H, packed_W = gt_raw.shape[1:]
        #to ensure the shape of the output of the model is same as gt_rgb
        gt_rgb = gt_rgb[:, :3*packed_H, :3*packed_W]

        if self.data_type == 'test' and self.merge_test:
            input_raw = self.crop_vertical_patch(input_raw)#tow numpy with shape (9,1344,1005)
            input_raw = [
                (np.float32(t) - self.black_level) / np.float32(self.white_level - self.black_level)
                for t in input_raw
            ]

            gt_raw = (np.float32(gt_raw) - self.black_level) / np.float32(self.white_level - self.black_level)
            gt_rgb = np.float32(gt_rgb) / np.float32(65535)


            if self.ratio:
                input_raw = [t * info['ratio'] for t in input_raw]
            if self.max_clip is not None:
                input_raw = [np.minimum(t, self.max_clip) for t in input_raw]
            if self.min_clip is not None:
                input_raw = [np.maximum(t, self.min_clip) for t in input_raw]

            gt_rgb = gt_rgb.clip(0.0, 1.0)

            input_raw = [torch.from_numpy(t).float() for t in input_raw]
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
        else:
            if self.patch_size:
                input_patch, gt_raw_patch, gt_rgb_patch = self.crop_random_patch(input_raw, gt_raw, gt_rgb, self.patch_size)
                if self.h_flip and np.random.randint(0,2) == 1 and self.data_type == 'train':  # random horizontal flip
                    input_patch = np.flip(input_patch, axis=2)
                    gt_rgb_patch = np.flip(gt_rgb_patch, axis=2)
                    gt_raw_patch = np.flip(gt_raw_patch, axis=2)
                if self.v_flip and np.random.randint(0,2) == 1 and self.data_type == 'train':  # random vertical flip
                    input_patch = np.flip(input_patch, axis=1)
                    gt_rgb_patch = np.flip(gt_rgb_patch, axis=1)
                    gt_raw_patch = np.flip(gt_raw_patch, axis=1)
                if self.transpose and np.random.randint(0,2) == 1 and self.data_type == 'train':  # random transpose
                    input_patch = np.transpose(input_patch, (0, 2, 1))
                    gt_rgb_patch = np.transpose(gt_rgb_patch, (0, 2, 1))
                    gt_raw_patch = np.transpose(gt_raw_patch, (0, 2, 1)) 

                input_raw = input_patch.copy()
                gt_raw = gt_raw_patch.copy()
                gt_rgb = gt_rgb_patch.copy()
            

            input_raw = (np.float32(input_raw) - self.black_level) / np.float32(self.white_level - self.black_level)
            gt_raw = (np.float32(gt_raw) - self.black_level) / np.float32(self.white_level - self.black_level)
            gt_rgb = np.float32(gt_rgb) / np.float32(65535)

            if self.ratio:
                input_raw = input_raw * info['ratio']
            if self.max_clip is not None:
                input_raw = np.minimum(input_raw, self.max_clip)
            if self.min_clip is not None:
                input_raw = np.maximum(input_raw, self.min_clip)

            gt_rgb = gt_rgb.clip(0.0, 1.0)

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

    def crop_vertical_patch(self, input_raw):
        '''
        input_raw: numpy with shape (9,1344,2010)
        output: list of numpy with shape (9,1344,1005)
        '''
        _, H, W = input_raw.shape
        input_raws = [input_raw[:,:,0:W//2],input_raw[:,:,W//2:]]
        return input_raws

if __name__=='__main__':
    seed = 3407
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    '''
    def combine_vertical_patches(patch1, patch2, original_width):
        assert patch1.shape[2] + patch2.shape[2] == original_width, "Widths of patches do not match the original width."
        combined_raw = np.concatenate((patch1, patch2), axis=2)
        return combined_raw

    dataset = SIDSonyDataset(data_dir='E:/Deep Learning/datasets/SID/Sony',
                            image_list_file='Sony_test_list.txt',data_type='test',merge_test=True)
    data = dataset[7]
    input_raws = data['input_raw']
    for t in input_raws:
        print(t.shape, t.min(), t.max())
    combined_image = combine_vertical_patches(input_raws[0], input_raws[1], 2128)
    print(combined_image.shape)
    '''
    dataset = SIDFujiDataset(data_dir='E:/Deep Learning/datasets/SID/Fuji',
                            image_list_file='Fuji_train_list.txt',data_type='train',merge_test=True,patch_size=384)
    data = dataset[7]
    input_raw, gt_raw, gt_rgb = data['input_raw'], data['gt_raw'], data['gt_rgb']
    print(input_raw.shape, gt_raw.shape, gt_rgb.shape)
    #print(len(input_raw), gt_raw.shape, gt_rgb.shape)
    #for i in range(len(input_raw)):
    #    print(input_raw[i].shape)   
    print(input_raw.min(),input_raw.max())
    print(gt_raw.min(),gt_raw.max())
    print(gt_rgb.min(),gt_rgb.max())

