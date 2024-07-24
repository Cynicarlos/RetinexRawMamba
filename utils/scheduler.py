from timm.scheduler.cosine_lr import CosineLRScheduler

def build_scheduler(config, optimizer):
    num_steps = int(config['last_lr_epoch'])
    warmup_steps = int(config['warmup_epochs'])
    lr_scheduler = None
    if config['lr_scheduler']['type'] == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            cycle_mul=1.,
            lr_min=config['min_lr'],
            warmup_lr_init=config.get('warmup_lr', 0.0),
            warmup_t=warmup_steps,
            cycle_limit=1
        )
    else:
        raise NotImplementedError()

    return lr_scheduler