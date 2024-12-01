import torch
import os
from collections import OrderedDict
ROOT = '/home/hsu/PycharmProjects/CTRegression'


def save_model(model, name, config):
    exp_name = config['ckpt_dir_name']
    path = f'{ROOT}/static/run/{exp_name}'
    if not os.path.exists(path):
        _path = path.split("/")
        check_path(f'/{_path[1]}', _path[2:])
    # print(f'target:{path}')
    if isinstance(model, OrderedDict):
        state_dict = model
    else:
        state_dict = model.state_dict()


    torch.save(state_dict, f'{path}/{name}.pth')
    
    
def check_path(path, param: list):
    # print(f'Current path:{path}')
    if not os.path.exists(path):
        os.mkdir(path)
        
    if len(param) > 0:
        check_path(f'{path}/{param.pop(0)}', param)
    return None