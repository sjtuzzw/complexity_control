from .composition import *
from .simple_task import *
from .chain_search import *
from .composition_random_data_complexity import *
from .composition_test import *
from .composition_more_anchor import *




def gen_sequence_group(args, mode, data_size, **kwargs):

    if args.target == "3x_to_x":
        seq_list = task_3x_to_x(args, mode, data_size)
        
    if args.target == "composition":
        seq_list = task_composition(args, mode, data_size)

    if args.target == "composition_test":
        seq_list = task_composition_test(args, mode, data_size)
    
    if args.target == "composition_random":
        seq_list = task_composition_random(args, mode, data_size)
    
    if args.target == "composition_more_anchor":
        seq_list = task_composition_more_anchor(args, mode, data_size)

    if args.target == "chain_search":
        seq_list = task_chain(args, mode, data_size)

    if args.target == "single_chain_search":
        seq_list = task_single_chain(args, mode, data_size)

    if args.target == "single_chain_search_with_order":
        seq_list = task_single_chain_with_order(args, mode, data_size)

    if args.target == "noised_double_chain_search":
        seq_list = task_noised_double_chain(args, mode, data_size)

    if args.target == "double_chain_search":
        seq_list = task_double_chain(args, mode, data_size)

    return seq_list