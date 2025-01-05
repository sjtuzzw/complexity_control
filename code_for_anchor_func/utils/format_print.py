def format_print(seq, word_color_dict):

    color_index_dict = {
        'r': '\033[31m',
        'g': '\033[32m',
        'y': '\033[33m',
        'b': '\033[34m',
        'purple': '\033[35m',
        'cyan': '\033[36m',
        'w': '\033[37m',
        'reset': '\033[0m',
        'r_bg': '\033[0;30;41m',      
        'g_bg': '\033[0;30;42m',      
        'y_bg': '\033[0;30;43m',      
        'b_bg': '\033[0;30;44m',      
        'purple_bg': '\033[0;30;45m', 
        'cyan_bg': '\033[0;30;46m',   
        'w_bg': '\033[0;30;47m',      
    }

    if isinstance(seq, str):
        if ', ' in seq:
            seq = seq.split(', ')
        elif ',' in seq:
            seq = seq.split(',')
        else:
            seq = seq.split(' ')
        
    for word in seq:
        if word in word_color_dict:
            print(color_index_dict[word_color_dict[word]], word, color_index_dict['reset'], end='')
        else:
            print(word, end=' ')
    print()