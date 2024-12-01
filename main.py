from npo.beato import dataset as NBD
from npo.beato import io as BIO
import prepare as P
import json
import datetime as dt
import os
import torch
import random
import numpy as np
from accelerate import Accelerator
ROOT = '/home/hsu/PycharmProjects/CTRegression'


def setting_seeds(config):
    rs = config['random_seed']
    random.seed(rs)
    np.random.seed(rs)
    torch.manual_seed(rs)
    torch.cuda.manual_seed(rs)


def save_config2exp(config):
    tpath = config['ckpt_dir_name']
    
    with open(f'{ROOT}/static/run/{tpath}/config.json', 'w+') as jout:
        json.dump(config, jout)
    

def load_config():
    # global ROOT
    with open(f'{ROOT}/static/config.json') as jin:
        config = json.load(jin)
    # config['loader']['collate_fn'] = eval(config['loader']['collate_fn'])
    config['now'] = f'{dt.datetime.now()}'    
    
    cnt = 0
    ckpt_name = f'exp0'
    while os.path.exists(f'{ROOT}/static/run/{ckpt_name}'):
        cnt += 1
        ckpt_name = f'exp{cnt}'    
        
    path = f'{ROOT}/static/run/{ckpt_name}'
    os.mkdir(path)
    config['ckpt_dir_name'] = ckpt_name
    logd = config['logging_dir']
    config['logging_dir'] = f'{ROOT}/static/log/{ckpt_name}'
    return config


def main(config, do_tracker=False):
    
    if config is None:
        config = load_config()
    setting_seeds(config)
    config['do_tracker'] = do_tracker
    # accelerator = Accelerator(split_batches=True)
    pack = P.prepare_all(config)
    model = pack['model']
    tloader = pack['tloader']
    vloader = pack['vloader']
    accelerator: Accelerator = pack['accelerator']
    optimizer = pack['optimizer']
    loss_fn = pack['loss_fn']
    warmup = pack['warmup']
    schedulers = pack['scheduler']
    # model, tloader, vloader, accelerator, optimizer, lr_scheduler, loss_fn = P.prepare_all(config)
    n_epochs = config['epoch']
    best_loss = 1e+18
    n_iter = len(tloader)

    for epoch in range(n_epochs):
        tloss = P.train_epoch(model, tloader, accelerator, optimizer, warmup, loss_fn, config, epochs=epoch)
        vloss = P.val(model, vloader, loss_fn, acc=accelerator)
        
        if n_iter * (epoch + 1) > config['warmup']['steps'] or not config['warmup']['do_warmup']:
            for pack in schedulers:
                pack[0].step(pack[1](vloss))
        # lr_scheduler.step()
        name = f'model-{epoch}'
        if not accelerator.is_local_main_process:
            continue
        # else:
            # accelerator.wait_for_everyone()
        
        if do_tracker:
            accelerator.log({                
                'val loss': vloss,         
                'train loss': tloss
            }, step=n_iter * (epoch + 1))
        
        if best_loss > vloss and accelerator.is_local_main_process:
            best_loss = vloss       
            config['best'] = vloss
            # accelerator.wait_for_everyone()
            BIO.save_model(accelerator.unwrap_model(model), 'best', config)
            continue
        
        if (epoch + 1) % 10 == 0 and accelerator.is_local_main_process:
            # accelerator.wait_for_everyone()
            BIO.save_model(accelerator.unwrap_model(model), name, config)
            continue
    # accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        BIO.save_model(accelerator.unwrap_model(model), 'last', config)
    # save_config2exp(config)
    

if __name__ == '__main__':
    # torch.cuda.memory._record_memory_history()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    import pynvml

    pynvml.nvmlInit()
    dc = pynvml.nvmlDeviceGetCount()
    handlers = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(dc)]
    usage_list = []    
    while True:
        
        for handler in handlers:            
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handler)
            mem_tot = mem_info.total
            mem_used = mem_info.used
            usage_list.append((mem_used / mem_tot) < .1)
        if all(usage_list):
            break
        usage_list.clear()

    pynvml.nvmlShutdown()
    config = load_config()
    try:
        main(config, True)
        config['status'] = 'success'
    except Exception as e:
        print(e.args)
        config['status'] = 'failed'
        config['message'] = e.args
    finally:
        save_config2exp(config)
        # continue
    # torch.cuda.memory._dump_snapshot('./test2.pickle')
    