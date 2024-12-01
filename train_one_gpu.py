from npo.beato import dataset as NBD
from npo.beato import io as BIO
from npo.beato import factory
from engine import get_method
import utils as BU
import prepare as P
import json
import datetime as dt
import os
import torch
import random
import traceback
import numpy as np
from accelerate import Accelerator
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
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
    with open(f'{ROOT}/static/config_one_gpu.json') as jin:
        config = json.load(jin)

    return config


def auto_fix_config(config, accelerator: Accelerator):
    config['now'] = f'{dt.datetime.now()}'

    cnt = 0
    ckpt_name = f'exp0'
    someone_done = False
    
    while os.path.exists(f'{ROOT}/static/run/{ckpt_name}'):
        cnt += 1
        ckpt_name = f'exp{cnt}'
        
        if someone_done:
            break

    someone_done = True

    path = f'{ROOT}/static/run/{ckpt_name}'
    if accelerator.is_local_main_process:
        os.mkdir(path)
    config['ckpt_dir_name'] = ckpt_name
    config['logging_dir'] = path
    return config


def main(config, accelerator: Accelerator, do_tracker=False):
    # with accelerator.local_main_process_first():
    config = auto_fix_config(config, accelerator)
    setting_seeds(config)
    config['do_tracker'] = do_tracker
    item_pack = factory.build(config, accelerator)

    train_epoch = get_method('train', config.get('num_gpu', 1) > 1 or config.get('dispatch', False))
    val = get_method('val', config.get('num_gpu', 1) > 1 or config.get('dispatch', False))
    n_epochs = config['epoch']
    best_loss = 1e+18
    n_iter = item_pack.n_iter('train')
    accelerator.print(f'Device: {accelerator.device}')
    leave_list = []

    for epoch in range(n_epochs):
        with accelerator.autocast():
            tloss, leave_list = train_epoch(
                item_pack, config,
                epochs=epoch, leave_list=leave_list
            )
            vloss, leave_list = val(item_pack, config, leave_list=leave_list)

        for pack in item_pack.scheduler:
            scheduler, wrapper = pack
            scheduler.step(wrapper(vloss))
        name = f'model-{epoch}'

        if do_tracker and accelerator.is_local_main_process:
            accelerator.log({
                'val loss': vloss,
                'train loss': tloss
            }, step=n_iter * (epoch + 1))

        if best_loss > vloss and accelerator.is_local_main_process:
            best_loss = vloss
            config['best'] = {
                'vloss': vloss,
                'epoch': epoch
            }
            # accelerator.wait_for_everyone()
            BIO.save_model(accelerator.unwrap_model(item_pack.model), 'best', config)
            continue

        if (epoch + 1) % 10 == 0 and accelerator.is_local_main_process:
            # accelerator.wait_for_everyone()
            BIO.save_model(accelerator.unwrap_model(item_pack.model), name, config)
            continue

    if accelerator.is_local_main_process:
        BIO.save_model(accelerator.unwrap_model(item_pack.model).state_dict(), 'last', config)


if __name__ == '__main__':
    # torch.cuda.memory._record_memory_history()
    import datetime
    torch.cuda.memory._record_memory_history(True)
    # torch.distributed.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=60))
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    config = load_config()
    accelerator: Accelerator = factory.declare_accelerator(config)
    if accelerator.is_local_main_process:
        BU.show_config(config)

    try:
        main(config, accelerator, True)
        config['status'] = 'success'
    except Exception as e:

        print(traceback.format_exc())
        config['status'] = 'failed'
        config['message'] = e.args
    finally:
        save_config2exp(config)
        # torch.cuda.memory._dump_snapshot('./static/debug/cuda_usage.pickle')
        # torch.cuda.memory._record_memory_history(enabled=None)
        # continue
    # torch.cuda.memory._dump_snapshot('./test2.pickle')
