import torch.optim as optim
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR



def get_optimizer(model, args, **kwargs):
    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=args.eps, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2))
    elif args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optim == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, eps=args.eps, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2))
    
    if args.scheduler == 'GradualWarmupScheduler_CosineAnnealingLR':

        scheduler_cosine = CosineAnnealingLR(optimizer, T_max=int(args.optim_T_max*args.num_batches), eta_min=float(args.optim_eta_min))
        scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=float(args.optim_multiplier), total_epoch=int(args.optim_total_epoch*args.num_batches), after_scheduler=scheduler_cosine)
        
        scheduler = scheduler_warmup
    elif args.scheduler == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step*args.num_batches, gamma=args.lr_decay_rate)
    else:
        scheduler = None

    return optimizer, scheduler