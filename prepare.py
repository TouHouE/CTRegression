import transformers
from accelerate import Accelerator, dispatch_model
from accelerate.utils import LoggerType
import torch
from torch import nn
from torch.utils.data import random_split
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader, Subset
from npo.beato import dataset as NBD
from torchvision.models import vit_l_32, vit_h_14, resnet18
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split as tts
import numpy as np
import torchmetrics
import peft


def prepare_loss_fn(config):
    lconfig = config.get('loss', {'init': 'torch.nn.MSELoss', 'init_param': {}})
    fn = eval(lconfig['init'])(**lconfig['init_param'])
    return fn


def prepare_scheduler(optimizer, config):
    lr_configs = config['lr_scheduler']
    scheduler_list = []
    
    for lr_config in lr_configs:
        scheduler = eval(lr_config['init'])(optimizer, **lr_config['init_param'])
        warp_steps = eval(lr_config['warp_step'])
        scheduler_list.append([scheduler, warp_steps])
    return scheduler_list
    

def prepare_warmup(optimizer, config):
    lr_config = config['warmup']
    # tot_steps = lr_config['total_steps']
    warmup_factor = lr_config['factor']
    warmup_steps = lr_config['steps']
    func = lambda step: min(step / warmup_steps, 1.) * warmup_factor + (1 - warmup_factor)
    warmup_lr = LambdaLR(optimizer, lr_lambda=func)
    return warmup_lr


def prepare_model(config):
    from npo.beato.models import reg
    # config['model']
    model = reg.RegModel()
    return model


def prepare_optimizer(model: nn.Module, config):
    optimizer = SGD(model.parameters(), lr=config['lr'])
    # optimizer = AdamW(model.parameters(), lr=config['lr'])
    return optimizer


def prepare_loader(config: dict, acc: Accelerator = None):
    ds = NBD.get_ds(config, acc=acc)
    rs = config['random_seed']
    loader_config = config['loader']
    test_ratio = loader_config['test_ratio']
    tconfig = loader_config['train']
    vconfig = loader_config.get('val', None)
    fn = eval(loader_config.get('collate_fn', 'torch.utils.data.default_collate'))
    if vconfig is None:
        return DataLoader(ds, **loader_config['train'], collate_fn=fn)

    # if config['num_gpu'] == 1:
    idx_list = np.arange(len(ds))
    train_ids, val_ids = tts(idx_list, test_size=test_ratio, random_state=rs)


    tds, vds = Subset(ds, train_ids), Subset(ds, val_ids)
    tloader = DataLoader(tds, **tconfig, collate_fn=fn)
    vloader = DataLoader(vds, **vconfig, collate_fn=fn)
    return tloader, vloader




def prepare_all(config: dict):
    sp = config.get('split_batch', False)
    if config['do_tracker'] and not config['debug']:
        accelerator = Accelerator(split_batches=sp, log_with=['wandb'], **config['accelerator'])
        accelerator.init_trackers(config['project'], config)

    else:
        accelerator = Accelerator(split_batches=sp, **config['accelerator'])
        # pass
    
    tloader, vloader = prepare_loader(config, acc=accelerator)
    config['warmup']['steps'] = config['warmup']['epochs'] * len(tloader)
    model = prepare_model(config)
    optimizer = prepare_optimizer(model, config)
    warmup = prepare_warmup(optimizer, config)
    scheduler_list = prepare_scheduler(optimizer, config)
    # model = dispatch_model(model, device_map='auto')
    model, tloader, vloader, optimizer, warmup = accelerator.prepare(model, tloader, vloader, optimizer, warmup)
    for i, pack in enumerate(scheduler_list):
        scheduler_list[i][0] = accelerator.prepare(pack[0])
    loss_fn = prepare_loss_fn(config)
    mapper = {
        'model': model,
        'tloader': tloader,
        'vloader': vloader,
        'accelerator': accelerator,
        'optimizer': optimizer,
        'scheduler': scheduler_list,
        'warmup': warmup,
        'loss_fn': accelerator.prepare(loss_fn),
    }
    return mapper
    

def train_epoch(
        model,
        tloader,
        accelerator: Accelerator,
        optimizer,
        warmup, loss_fn, config, **kwargs):
    
    model.train()
    total = len(tloader)
    pbar = tqdm(enumerate(tloader), total=total, ncols=80, disable=not accelerator.is_local_main_process)
    loss_train = .0
    ep = kwargs['epochs']
    pbar.set_description(f'Epochs:[{ep}]:')
    
    # with open('./static/wtf.txt', 'a+') as fout:
    for idx, pack in pbar:
        with accelerator.accumulate(model):
            if config['warmup']['do_warmup']:
                warmup.step()
            optimizer.zero_grad(set_to_none=True)
            pred_ = model(pack['image'])        
            pred = torch.sum(pred_, dim=0)     
            # with accelerator.autocast(cache_enabled=True):
            loss_iter = loss_fn(torch.log(pred + 1) / torch.log(torch.e), pack['label'][:1])
            accelerator.backward(loss_iter)
            has_nan = any(torch.isnan(param.grad).any() for param in model.parameters())

            # Clip the gradients to a maximum absolute value of 1.0
            if has_nan:
                # accelerator.print("Gradient contains NaN. Clipping gradient values...")
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=.5)
            optimizer.step()

        with torch.no_grad():
            loss_train += loss_iter.item()
            pbar.set_postfix({'train loss': loss_train / (idx + 1)})
            accelerator.log({
                'step loss': loss_train / (idx + 1),
                'lr': optimizer.param_groups[0]['lr']
            }, step=ep * total + (idx + 1))
        # break
    # exit(0)
    return loss_train / (idx + 1)


@torch.no_grad()
def val(model, vloader, loss_fn, acc, **kwargs):
    model.eval()
    pbar = tqdm(enumerate(vloader), total=len(vloader), ncols=80, disable=not acc.is_local_main_process)
    loss_val = .0

    for idx, pack in pbar:
        pred = model(pack['image'])
        pred = torch.sum(pred, dim=0)
        loss_iter = loss_fn(pred, pack['label'])
        
        loss_val += loss_iter.item()
        pbar.set_postfix({'val loss': loss_val / (idx + 1)})

    return loss_val / (idx + 1)
