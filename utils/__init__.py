import os
import random
import shutil
import torch
import numpy as np

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def load_checkpoint(config, checkpoint_path, model, optimizer, lr_scheduler, logger, epoch=None):
    logger.info(f"==============> Resuming form {checkpoint_path}....................")
    if checkpoint_path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            checkpoint_path, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_psnr = 0.0
    if not config.get('eval_mode', False) and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    if 'max_psnr' in checkpoint:
        max_psnr = checkpoint['max_psnr']
    if epoch is None and 'epoch' in checkpoint:
        config['train']['start_epoch'] = checkpoint['epoch']
        logger.info(f"=> loaded successfully '{checkpoint_path}' (epoch {checkpoint['epoch']})")
    del checkpoint
    torch.cuda.empty_cache()
    return max_psnr

def save_checkpoint(config, epoch, model, max_psnr, optimizer, lr_scheduler, is_best=False):
    save_state = {'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'max_psnr': max_psnr,
                'epoch': epoch,
                'config': config}

    os.makedirs(os.path.join(config['output'], 'checkpoints'), exist_ok=True)

    save_path = os.path.join(config['output'], 'checkpoints', 'checkpoint.pth')
    torch.save(save_state, save_path)
    if epoch % 100 == 0:
        shutil.copy(save_path, os.path.join(config['output'], 'checkpoints', f'epoch_{epoch:04d}.pth'))
    if is_best:
        shutil.copy(save_path, os.path.join(config['output'], 'checkpoints', 'best_model.pth'))

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm
