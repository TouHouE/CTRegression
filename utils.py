import json
import os
import re

def write_info(path, file, content, end='\n'):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        
    if isinstance(content, list):
        content = ', '.join(content)    
        
    with open(f'{path}/{file}', 'a+') as fout:
        fout.write(content + end)



def show_config(config: dict):
    _cfg = config.copy()
    
    for key, value in config.items():
        if isinstance(_cfg[key], dict):
            print(f'{key} Config:\n{json.dumps(value, indent=5)}')
            del _cfg[key]
    print(f"Common Config: \n{json.dumps(_cfg, indent=5)}")
    

def get_loader_info(loader, mode):
    record_list = []
    
    for i, pack in enumerate(loader):
        if (i + 1) % 256 == 0:
            record_list.append(pack['fname'])
    return record_list