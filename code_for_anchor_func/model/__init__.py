# from .choose_model import *
from .choose_optimizer import *

from .GPT2_init_for_diff_part_prenorm import myGPT2_init_for_diff_part_prenorm

def get_model(args, device, **kwargs):
    if args.model == 'GPT2_init_for_diff_part_prenorm':
        model = myGPT2_init_for_diff_part_prenorm(args, device).to(device)

    return model