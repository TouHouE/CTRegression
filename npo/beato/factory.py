from typing import Union, List, Callable, Any, Optional
import dataclasses
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.optim import Optimizer, SGD
from torch.optim.lr_scheduler import LRScheduler
import accelerate
from accelerate import Accelerator
from accelerate.utils import DistributedType
import numpy as np
from sklearn.model_selection import train_test_split as tts

from npo.beato import dataset as NBD


@dataclasses.dataclass
class ElementPack:
    model: nn.Module
    tloader: DataLoader
    optimizer: Optimizer
    loss_fn: nn.Module
    accelerator: Accelerator
    scheduler: Optional[List[Union[LRScheduler, Callable]]] = None
    vloader: Optional[DataLoader] = None

    def train_pack(self) -> dict:
        """
            Using key ("model", 'loader', 'optimizer', 'accelerator', 'scheduler' to access item.
        :return: A dictionary store all item training needed.
        """
        map = {
            'model': self.model,
            'loader': self.tloader,
            'loss_fn': self.loss_fn,
            'optimizer': self.optimizer,
            'accelerator': self.accelerator,
        }

        if self.scheduler is not None:
            map['scheduler'] = self.scheduler
        return map



    def val_pack(self):
        """
            with order: model, dataloader, loss function, accelerator
        :return: (nn.Module, DataLoader, Loss, accelerate.Accelerator)
        """
        map = {
            'model': self.model,
            'loader': self.vloader,
            'loss_fn': self.loss_fn,
            'accelerator': self.accelerator
        }
        return map

    def n_iter(self, loader_type:str = 'train'):
        return len(self.tloader) if loader_type.lower() == 'train' else len(self.vloader)


def declare_accelerator(config) -> Accelerator:
    sp = config.get('split_batch', False)
    aconfig = config['accelerator']
    do_log = config.get('tracker', False)
    from accelerate.utils import ProjectConfiguration
    pconfig = ProjectConfiguration(project_dir=config['logging_dir'])
    
    if do_log is not None and not config['debug']:
        accelerator = Accelerator(split_batches=sp, log_with=['wandb'], project_config=pconfig, **aconfig)
        accelerator.init_trackers(config['project'], config)
    else:
        accelerator = Accelerator(split_batches=sp, project_config=pconfig, **aconfig)
    return accelerator


def declare_loss_fn(config):
    lconfig = config.get('loss', {'init': 'torch.nn.MSELoss', 'init_param': {}})
    fn = eval(lconfig['init'])(**lconfig['init_param'])
    return fn


def declare_scheduler(optimizer, accelerator: Accelerator, config: dict):
    from accelerate.utils import DistributedType, MegatronLMDummyScheduler
    # accelerator.distributed_type
    if accelerator.distributed_type == DistributedType.MEGATRON_LM:
        scheduler = MegatronLMDummyScheduler(
            optimizer=optimizer,
            **config['warmup']
        )
        return [scheduler, lambda x: x]

    lr_configs = config['lr_scheduler']
    scheduler_list = []

    for lr_config in lr_configs:
        scheduler = eval(lr_config['init'])(optimizer, **lr_config['init_param'])
        warp_steps = eval(lr_config['warp_step'])
        scheduler_list.append([scheduler, warp_steps])
    return scheduler_list


def declare_model(accelerator: Accelerator, config: dict) -> nn.Module:
    from npo.beato.models import reg
    # config['model']
    model = reg.RegModel(backbone='vit_large_patch32_224')
    balanced_allocated = accelerate.utils.get_balanced_memory(
        model,
        max_memory={
            0: '20GB',
            1: '30GB',
            2: '30GB',
            # 3: '0GB',
            'cpu': '50GB'
        },
        no_split_module_classes=['EncoderBlock', 'Block']
    )
    device_map = accelerate.utils.infer_auto_device_map(
        model,
        balanced_allocated,
        no_split_module_classes=['EncoderBlock', 'Block']
    )
    hooker = accelerate.big_modeling.AlignDevicesHook(
        weights_map=device_map,
        execution_device='cuda:0',
        place_submodules=True,
    )
    accelerate.big_modeling.add_hook_to_module(model, hooker, append=True)
    return accelerate.dispatch_model(
        model,
        device_map=device_map,
        main_device='cuda:0',
        offload_dir=accelerator.project_dir
    )




    # return model


def declare_optimizer(model: nn.Module, config):
    from torch.optim import Adam
    # from accelerate.utils import DummyOptim
    # optimizer = A
    # optimizer = DummyOptim(model.parameters(), lr=config['lr'])
    
    optimizer = Adam(model.parameters(), lr=config['lr'])
    # optimizer = AdamW(model.parameters(), lr=config['lr'])
    return optimizer


def declare_loader(config: dict, acc: Accelerator = None):
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


def build(config, accelerator: Accelerator):
    model = declare_model(accelerator, config)
    optimizer = declare_optimizer(model, config)
    schedulers = declare_scheduler(optimizer, accelerator, config)
    loader_pack = declare_loader(config, accelerator)
    loss_fn = declare_loss_fn(config)
    if accelerator.distributed_type == DistributedType.MEGATRON_LM:
        model, optimizer, schedulers, tloader, vloader = accelerator.prepare(
            model, optimizer, schedulers, loader_pack[0], loader_pack[1]
        )
    else:
        model = accelerator.prepare(model)
        optimizer = accelerator.prepare(optimizer)
        schedulers = accelerator.prepare(*schedulers)
        # accelerator._prepare_megatron_lm()
    
        if isinstance(loader_pack, tuple):
            tloader, vloader = accelerator.prepare(*loader_pack)        
        else:
            tloader = accelerator.prepare(loader_pack)
            vloader = None

    pack = ElementPack(
        model=model,
        optimizer=optimizer,
        scheduler=schedulers,
        loss_fn=loss_fn,
        tloader=tloader,
        vloader=vloader,
        accelerator=accelerator
    )
    return pack
