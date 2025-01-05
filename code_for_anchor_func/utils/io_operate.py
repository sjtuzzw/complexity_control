import yaml
import json
import re
from _ctypes import PyObj_FromPtr
import numpy as np
import logging
import argparse



# ======================== yaml ========================


def read_yaml_data(yaml_file):
    file = open(yaml_file, 'r', encoding="utf-8")         
    data = yaml.load(file.read(), Loader=yaml.FullLoader) 
    file.close()
    return data

def write_yaml_data(yaml_file, data):
    file = open(yaml_file, 'w', encoding="utf-8")       
    yaml.dump(data, file, allow_unicode=True)
    file.close()




# ======================== args ========================

def load_args(args_path):

    args = read_json_data(args_path)
    args = argparse.Namespace(**args)

    return args



# ======================== json ========================


class NoIndent(object):
    def __init__(self, value):
        self.value = value

class MyEncoder(json.JSONEncoder):
    FORMAT_SPEC = '@@{}@@'
    regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))

    def __init__(self, **kwargs):
        # Save copy of any keyword argument values needed for use here.
        self.__sort_keys = kwargs.get('sort_keys', None)
        super(MyEncoder, self).__init__(**kwargs)

    def default(self, obj):
        return (self.FORMAT_SPEC.format(id(obj)) if isinstance(obj, NoIndent)
                else super(MyEncoder, self).default(obj))

    def encode(self, obj):
        format_spec = self.FORMAT_SPEC  # Local var to expedite access.
        json_repr = super(MyEncoder, self).encode(obj)  # Default JSON.

        # Replace any marked-up object ids in the JSON repr with the
        # value returned from the json.dumps() of the corresponding
        # wrapped Python object.
        for match in self.regex.finditer(json_repr):
            # see https://stackoverflow.com/a/15012814/355230
            id = int(match.group(1))
            no_indent = PyObj_FromPtr(id)
            json_obj_repr = json.dumps(no_indent.value, sort_keys=self.__sort_keys)

            # Replace the matched id string with json formatted representation
            # of the corresponding Python object.
            json_repr = json_repr.replace(
                            '"{}"'.format(format_spec.format(id)), json_obj_repr)

        return json_repr


def read_json_data(json_file_name):
    with open(json_file_name, 'r', encoding='utf8')as fp:
        args = json.load(fp)
    fp.close()
    return args

def write_json_data(json_file_name, args):
    with open(json_file_name, "w") as f:
        f.write(json.dumps(args, ensure_ascii=False, indent=4, separators=(',', ':')))
    f.close()

def save_to_json_noindent(datas: dict, json_save_path: str):

    for k, v in datas.items():
        if isinstance(v, dict) or isinstance(v, list):
            datas[k] = NoIndent(datas[k])
        if isinstance(v, np.ndarray):
            datas[k] = NoIndent(datas[k].tolist())

    with open(json_save_path, 'w') as fw:
        json_data = json.dumps(datas, cls=MyEncoder, ensure_ascii=False, sort_keys=False, indent=2)
        fw.write(json_data)
        fw.write('\n')

class Dict2Class(object):
	def __init__(self, entries: dict={}):
		for k, v in entries.items():
			if isinstance(v, dict):
				self.__dict__[k] = Dict2Class(v)
			else:
				self.__dict__[k] = v





# ======================== logging ========================


class Log:
    def __init__(self, file_name, mode = 'a'):
    
        self.logger = logging.getLogger(file_name)  
        self.logger.setLevel(logging.DEBUG)  

        fmt = "%(asctime)s - %(levelname)s: %(message)s"
        formatter = logging.Formatter(fmt)

        fh = logging.FileHandler(file_name, mode=mode)
        fh.setLevel(logging.DEBUG)  
        fh.setFormatter(formatter)

        self.logger.handlers = []
        self.logger.addHandler(fh)
        # self.logger.addHandler(sh)

    def info(self, message):
        self.logger.info(message)


def setup_logger(log_file, level=logging.INFO):
    l = logging.getLogger(log_file[:-4])
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fileHandler)