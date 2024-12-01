from typing import Callable
from accelerate import Accelerator
import accelerate
import torch
from torch import nn
from tqdm.auto import tqdm
import gc
import utils as BU
from npo.beato.factory import ElementPack
import traceback
from accelerate.utils import DistributedType

@accelerate.utils.find_executable_batch_size(starting_batch_size=1024)
def launch_a_iter(batch_size, model, image, accelerator: Accelerator):
    pred = 0
    dis_batch = torch.split(image.cpu(), batch_size)
    batch_bar = tqdm(dis_batch, total=len(dis_batch), desc=f'Batch({batch_size}) process', leave=False, ncols=100)
    try:
        for cimg in batch_bar:
            cimg = cimg.to(accelerator.device)
            shape: torch.Size = cimg.size()
            # shape.
            pred_ = torch.sum(model(cimg))
            # Poor cuda memory
            pred += pred_

            batch_bar.set_postfix({'Batch Agatston': pred_.item(), 'Size': list(shape)})
            # accelerator
            del cimg, pred_
            torch.cuda.empty_cache()
            gc.collect()
    except Exception as e:
        print(traceback.format_exc())
        return None
    return pred


def ds_iter(model, image, accelerator: Accelerator):
    # with accelerator.autocast():
    pred = model(image)
    return torch.sum(pred)
        

def multi_process_iter(model, image, accelerator: Accelerator):
    try:
        import math
        image_split = torch.split(image, math.ceil(image.size(0) / n_gpu))
    except Exception as e:
        BU.write_info(config['logging_dir'], 'SplitError.txt', f'{e.args}')
        leave_list.append(pack['file'])
        denominator_bias += 1
        return
    pred = .0

    with accelerator.split_between_processes(image_split) as sub_image:
        sub_image = sub_image[0]
        sub_pred = model(sub_image.to(accelerator.device))
        pred += torch.sum(sub_pred)
    return pred

def train_epoch(item_pack: ElementPack, config, **kwargs):
    tpack = item_pack.train_pack()
    model: nn.Module = tpack['model']
    tloader: torch.utils.data.DataLoader = tpack['loader']
    optimizer: torch.optim.Optimizer = tpack['optimizer']
    accelerator: Accelerator = tpack['accelerator']
    loss_fn: nn.Module = tpack['loss_fn']
    total = item_pack.n_iter('train')

    model.train()
    loss_train = .0
    ep = kwargs['epochs']
    leave_list: list[str] = kwargs.get('leave_list', [])
    pbar = tqdm(enumerate(tloader), total=total, ncols=140, disable=not accelerator.is_local_main_process, desc=f'Epochs:[{ep}]:')

    for idx, pack in pbar:
        clip_flag = False
        if pack['file'] in leave_list:
            continue
        print(f'Process-{accelerator.process_index} file: {pack["file"]}')
        optimizer.zero_grad(set_to_none=True)
        if accelerator.distributed_type == DistributedType.DEEPSPEED:
            # pred = ds_iter(model, pack['image'], accelerator)
            pred = torch.sum(model(pack['image'].to(torch.half)))
        else:
            pred = launch_a_iter(model, pack['image'], accelerator)
        if pred is None:
            BU.write_info(config['logging_dir'], 'out_cuda.txt', pack['file'])
            leave_list.append(pack['file'])
            accelerator.print(f'Got Error')
            continue

        loss_iter = loss_fn(pred, torch.log(pack['label'] + 1).to(accelerator.device).squeeze(0))
        accelerator.backward(loss_iter)
        accelerator.clip_grad_norm_(model.parameters(), max_norm=1.4)
        optimizer.step()

        if loss_iter.item() / 100 > loss_train / (idx + 1):
            BU.write_info(config['logging_dir'], 'outlier.txt',
                          f'Outlier: {loss_iter.item():15.5f}, File: {pack["file"]}')

        with torch.no_grad():
            loss_train += loss_iter.item()
            pbar.set_postfix(
                {
                    'Iter loss': loss_iter.item(),
                    'Train loss': loss_train / (idx + 1),
                    'Agatston': pred.item(),                    
                    'File': pack['file']
                }
            )

            if not config['debug']:
                accelerator.log({
                    'Train iter loss': loss_iter.item(),
                    'Train Accumulation Loss': loss_train / (idx + 1),
                    'Train Agatston': pred.item(),
                    'Train Agatston Label(log)': torch.log(pack['label'] + 1).item(),
                    'Train Agatston Label': pack['label'].item(),
                    'lr': optimizer.param_groups[0]['lr']
                }, step=ep * total + (idx + 1))
    return loss_train / (idx + 1), leave_list


@torch.no_grad()
def val(item_pack: ElementPack, config, **kwargs):
    val_pack = item_pack.val_pack()
    model = val_pack['model']
    vloader = val_pack['loader']
    acc = val_pack['accelerator']
    loss_fn = val_pack['loss_fn']
    leave_list = kwargs.get('leave_list', [])
    
    model.eval()
    loss_val = .0
    pbar = tqdm(enumerate(vloader), total=len(vloader), ncols=120, disable=not acc.is_local_main_process)

    for idx, pack in pbar:
        # try:
        if pack['file'] in leave_list:
            continue
        pred = launch_a_iter(model, pack['image'], acc)
        if pred is None:
            leave_list.extend([pack['file']] if isinstance(pack['file'], str) else pack['file'])
            continue
        loss_iter = loss_fn(pred, torch.log(pack['label'] + 1).squeeze(0).to(acc.device))

        loss_val += loss_iter.item()
        pbar.set_postfix({'val loss': loss_val / (idx + 1)})

    return loss_val / (idx + 1), leave_list


def train_on_multi_gpu(
        item_pack: ElementPack, config, **kwargs
):
    train_pack = item_pack.train_pack()
    model = train_pack['model']
    tloader = train_pack['loader']
    accelerator = train_pack['accelerator']
    optimizer = train_pack['optimizer']
    loss_fn = train_pack['loss_fn']

    model.train()
    total = len(tloader)
    pbar = tqdm(enumerate(tloader), total=total, ncols=120, disable=not accelerator.is_local_main_process)
    loss_train = .0
    n_gpu = accelerator.num_processes
    ep = kwargs['epochs']
    leave_list: list[str] = kwargs.get('leave_list', [])
    pbar.set_description(f'Epochs:[{ep}]:')
    denominator_bias = 0

    for idx, pack in pbar:
        clip_flag = False
        if pack['file'] in leave_list:
            denominator_bias += 1
            continue
        image = pack['image']

        optimizer.zero_grad(set_to_none=True)
        if accelerator.num_processes > 1:
            pred = multi_process_iter(model, image, accelerator)
            accelerator.wait_for_everyone()
        else:
            pred = torch.sum(model(image))    
        
        loss_iter = loss_fn(pred, torch.log(pack['label'] + 1).to(accelerator.device).squeeze(0))
        accelerator.backward(loss_iter)
        accelerator.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        if loss_iter.item() / 100 > loss_train / (idx + 1 - denominator_bias):
            BU.write_info(config['logging_dir'], 'outlier.txt',
                          f'Outlier: {loss_iter.item():15.5f}, File: {pack["file"]}')

        with torch.no_grad():
            loss_train += loss_iter.item()
            pbar.set_postfix(
                {
                    'Iter loss': loss_iter.item(),
                    'Train loss': loss_train / (idx + 1 - denominator_bias),
                    'Agatston': pred.item(),
                    'Do clip': clip_flag
                }
            )

            if not config['debug']:
                accelerator.log({
                    'Train iter loss': loss_iter.item(),
                    'Train Accumulation Loss': loss_train / (idx + 1 - denominator_bias),
                    'Train Agatston': pred.item(),
                    'Train Agatston Label(log)': torch.log(pack['label'] + 1).item(),
                    'Train Agatston Label': pack['label'].item(),
                    'lr': optimizer.param_groups[0]['lr']
                }, step=ep * total + (idx + 1))
    return loss_train / (idx + 1 - denominator_bias), leave_list


@torch.no_grad()
def val_on_multi_gpu(
        item_pack: ElementPack, config, **kwargs
):
    # Access validation items from ElementPack. #
    val_pack = item_pack.val_pack()
    model = val_pack['model']
    acc = val_pack['accelerator']
    vloader = val_pack['loader']
    loss_fn = val_pack['loss_fn']
    # Access Done. #

    model.eval()
    loss_val = .0
    denominator_bias = 0
    n_gpu = acc.num_processes
    leave_list = kwargs.get('leave_list', [])
    pbar = tqdm(enumerate(vloader), total=len(vloader), ncols=120, disable=not acc.is_local_main_process)

    for idx, pack in pbar:
        if pack['file'] in leave_list:
            denominator_bias += 1
            continue
        image = pack['image']

        if acc.num_processes > 1:
            pred = multi_process_iter(model, image, acc)
            accelerator.wait_for_everyone()
        else:
            pred = torch.sum(model(image))

        loss_iter = loss_fn(pred, torch.log(pack['label'] + 1).squeeze(0).to(acc.device))

        loss_val += loss_iter.item()
        pbar.set_postfix({'val loss': loss_val / (idx + 1 - denominator_bias)})

    return loss_val / (idx + 1 - denominator_bias), leave_list


METHOD_POOL = {
    'train': {
        True: train_on_multi_gpu,
        False: train_epoch
    },
    'val': {
        True: val_on_multi_gpu,
        False: val
    }
}


def get_method(phase: str, is_mult_gpu=True) -> Callable:
    method: Callable = METHOD_POOL.get(phase.lower(), METHOD_POOL['val'])[is_mult_gpu]
    print(f'Selected method: {method.__name__}')
    return method
    
