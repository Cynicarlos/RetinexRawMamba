import argparse
import numpy as np
import os
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from datasets import build_test_loader
from models import build_model
from utils import crop_tow_patch, set_random_seed
from utils.metrics import get_psnr_torch, get_ssim_torch, get_lpips_torch

@torch.no_grad()
def test(model, dataloader, merge_test=False, save_image=False, save_dir=None):
    if save_image: 
        assert save_dir is not None

    model.eval()
    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    total_samples = len(dataloader.dataset)
    tqdm_loader = tqdm(dataloader, desc="Evaluating", leave=False)

    with open('Fuji_results.txt', 'a') as f:
        for idx, data in enumerate(tqdm_loader):
            input_path, gt_path, ratio = data['input_path'][0], data['gt_path'][0], data['ratio'][0]
            input_raw = data['input_raw'].cuda(non_blocking=True)
            gt_rgb = data['gt_rgb'].cuda(non_blocking=True)
            
            if merge_test:
                inputs = crop_tow_patch(input_raw)
                preds_1 = model(inputs[0])
                preds_2 = model(inputs[1])
                preds = [torch.cat([preds_1[i], preds_2[i]], dim=-1) for i in range(2)]
            else:
                preds = model(input_raw)#pred_rgb, pred_raw

            pred_rgb = preds[0]
            
            pred_rgb = torch.clamp(pred_rgb, 0, 1)
            gt_rgb = torch.clamp(gt_rgb, 0, 1)
            
            lpips = get_lpips_torch(pred_rgb, gt_rgb)

            pred_rgb =  pred_rgb * 255
            gt_rgb = gt_rgb * 255
            pred_rgb = pred_rgb.round()
            gt_rgb = gt_rgb.round()

            psnr = get_psnr_torch(pred_rgb, gt_rgb)
            ssim = get_ssim_torch(pred_rgb, gt_rgb)

            f.write(f"input:{input_path}    gt:{gt_path}    psnr:{psnr.item():.4f}   ssim:{ssim.item():.4f}  lpips:{lpips.item():.4f} ratio:{ratio}\n")

            psnr_sum += psnr.item()
            ssim_sum += ssim.item()
            lpips_sum += lpips.item()
            tqdm_loader.set_postfix({'psnr':f'{psnr.item():.4f}','ssim':f'{ssim.item():.4f}', 'lpips':f'{lpips.item():.4f}', 'avg_psnr': f'{psnr_sum/(idx+1):.4f}','avg_ssim':f'{ssim_sum/(idx+1):.4f}', 'avg_lpips': f'{lpips_sum/(idx+1):.4f}'}, refresh=True)
            if save_image:
                pred_rgb = pred_rgb.squeeze(0).cpu().numpy().astype(np.uint8)
                save_path = os.path.join(save_dir, os.path.basename(input_path)[:-4]) + '.jpg'
                Image.fromarray(pred_rgb.transpose(1, 2, 0)).save(save_path)
        average_psnr = psnr_sum / total_samples
        average_ssim = ssim_sum / total_samples
        average_lpips = lpips_sum / total_samples
        print(f'psnr:{average_psnr:.4f}     ssim:{average_ssim:.4f}       lpips:{average_lpips:.4f}')
        f.write(f'psnr:{average_psnr:.4f}       ssim:{average_ssim:.4f}     lpips:{average_lpips:.4f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--merge_test', action='store_true', default=False)
    args = parser.parse_args()
    
    with open('configs/fuji.yaml', 'r') as file:
        config = yaml.safe_load(file)
    set_random_seed(config['manual_seed'])

    model_name, model = build_model(config['model'])
    model = model.cuda()
    checkpoint = torch.load('pretrained/fuji_best_model.pth')
    print(f"epoch: {checkpoint['epoch']}")
    model.load_state_dict(checkpoint['model'])

    test_dataloader = build_test_loader(config['data'])
    test(model, test_dataloader, merge_test=args.merge_test, save_image=True, save_dir=save_dir)
    #save_dir = f"./visualization/{model_name}/Fuji_results"
    #os.makedirs(save_dir,exist_ok=True)
    #test(model, test_dataloader, merge_test=args.merge_test, save_image=False, save_dir=None)
    
    
    
