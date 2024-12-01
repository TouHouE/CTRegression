from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torch
import pandas as pd
import numpy as np
# import monai
import nibabel as nib
import os
import random
import json
import re
from accelerate import Accelerator

random.seed(114514)
LOC = ['LM', 'LAD', 'LCX', 'RCA', 'TotalArtery']
ALL_LABEL_KEY = [f'選中的CCTA報告::鈣化分數{name}_a' for name in LOC]
PATIENT_ID = '匿名化後ID'

def default_T():
    T.ToTensor()
    comp = T.Compose(
        [T.ToTensor()]
    )
    return comp

def get_mapper(label_path):
    df = pd.read_excel(label_path)
    # Convert Mandarin columns name into English
    df.rename(columns={ok: nk for nk, ok in zip(LOC, ALL_LABEL_KEY)}, inplace=True)
    # Remove all alphabetic character in patient name ex: "CVAI-0001" into "0001"
    patient_name: list = df[PATIENT_ID].apply(lambda x: str(x.split('-')[-1])).tolist()
    df[PATIENT_ID] = patient_name  # Replace original patient name
    mapper = df[LOC + [PATIENT_ID]].set_index(PATIENT_ID).to_dict()
    return mapper


def create_agtston_score_map(label_path, image_path, target):
    df = pd.read_excel(label_path)
    ds_map = {pidx: {'image': [], 'meta': [], 'label': []} for pidx in self.patient_name}

    for patient_name in self.patient_name.keys():
        patient_folder = f'{image_path}/{patient_name}'
        whole_file = os.listdir(patient_folder)
        ds_map[patient_name]['image'].extend(list(filter(lambda x: x.endswith('.nii'), whole_file)))
        ds_map[patient_name]['meta'].extend(list(filter(lambda x: x.endswith('.json'), whole_file)))

    ds_map['label'].extend(df[target].astype(np.float32).tolist())


def collate_fn_no_meta(batch):
    # print(batch)
    images, labels, file = zip(*batch)


    # return [{'image': img, 'label': lab, 'meta': met} for img, lab, met in zip(images, labels, metas)]
    return {
        'image': torch.stack(images, dim=1),
        'label': torch.cat(labels),
        'file': file
    }


def collate_fn(batch):
    images, labels, _metas = zip(*batch)
    metas = {k: [meta[k] for meta in _metas] for k in _metas[0]}

    # return [{'image': img, 'label': lab, 'meta': met} for img, lab, met in zip(images, labels, metas)]
    return {
        'image': torch.stack(images, dim=0),
        'label': torch.cat(labels[:1]),
        'meta': metas
    }


class AgatstonScoreDataset(Dataset):
    def __init__(self, label_path, image_path, target, pseudo_batch_size=16, acc: Accelerator=None, multi_gpu=True):
        """

        :param label_path:
        :param image_path:
        :param target: belongs to :LOC
        """
        self.acc: Accelerator = acc
        self.img_buf = None
        self.multi_gpu = multi_gpu
        self.bs = pseudo_batch_size
        df = pd.read_excel(label_path)
        self.readable_nii = pd.read_csv('./static/len_ct.csv')
        # Convert Mandarin columns name into English
        df.rename(columns={ok: nk for nk, ok in zip(LOC, ALL_LABEL_KEY)}, inplace=True)
        # Remove all alphabetic character in patient name ex: "CVAI-0001" into "0001"
        patient_name: list = df[PATIENT_ID].apply(lambda x: str(x.split('-')[-1])).tolist()
        df[PATIENT_ID] = patient_name  # Replace original patient name
        self.mapper = df[LOC + [PATIENT_ID]].set_index(PATIENT_ID).to_dict()

        self.image_root = image_path
        # using patient name as key to get corresponding Agatston score.

        self.image_path_list = []
        self.patient_name_list = []
        self.target = target

        for patient_name_ in patient_name:
            patient_folder = f'{image_path}/{patient_name_}'
            # This statement is using to prevent calling the data is transmitting.
            if not os.path.exists(patient_folder):
                patient_name.remove(patient_name_)
                continue

            tmp = []
            # Readable file checking
            for name in os.listdir(patient_folder):
                nii_path = f'{patient_folder}/{name}'
                try:
                    pack = nib.load(nii_path)
                    pack.get_fdata()
                except Exception as E:
                    # self.acc.print(f'Error loading {nii_path}')
                    continue
                tmp.append(nii_path)
            # Readable check Done.
            # tmp = [f'{patient_folder}/{name}' for name in os.listdir(patient_folder) if name.endswith('.nii')]
            # Because each patient has at least one CT-series, thus need to iter the patient folder.
            self.image_path_list.extend(tmp)
            self.patient_name_list.extend([patient_name_] * len(tmp))
            if len(self.image_path_list) > 500:
                break

        self.acc.print(f'Initial Dataset OK')
        
        # self.acc.print(self.mapper)

    def __len__(self):
        if self.multi_gpu:
            return len(self.image_path_list) * 256
        return len(self.image_path_list)

    def __getitem__(self, index: int):
        if self.multi_gpu:
            return self._get_multi_gpu(index)
        return self._get_single_gpu(index)

    def _get_multi_gpu(self, index: int):
        # with self.acc.on_local_main_process()
        if self.img_buf is None:
            return self._first_get(index)
        z_idx = index % 256        
        # self.acc.print(f'{self.acc.local_process_index}Index: {index}, slice: {z_idx}')
        img, label, ometa = self.img_buf
        if z_idx == 255:
            self.img_buf = None
            self.acc.print(f'Clean buf')
            
        if img.shape[0] - 1 < z_idx:
            img = np.zeros(img.shape[1:])
        else:
            img = img[z_idx, ...]
        img, label = torch.from_numpy(img).float(), torch.as_tensor([label], dtype=torch.float32)
        meta = ometa.copy()
        meta['idx'] = z_idx
        return img, label, meta

    def _first_get(self, index: int):
        img, label, meta = self._first_process(index)        
        
        if index < img.shape[0]:
            img = img[index, ...]
        else:
            img = np.zeros(img.shape[1:])
                        
        img, label = torch.from_numpy(img).float(), torch.as_tensor([label], dtype=torch.float32)
        return img, label, meta

    def _first_process(self, index):
        item = index // 256 if self.multi_gpu else index
        patient_name = self.patient_name_list[item]
        image_path = self.image_path_list[item]
        label = self.mapper[self.target][patient_name] + 1e-6
        pack = nib.load(image_path)
        img: np.ndarray = pack.get_fdata()  # Expected shape: (H, W, D), but it might be: (H, W, D, ?)
        # self.img_buf = [pack, label, img]

        if len(img.shape) > 3:  # Some .nii image get over 3 dimension with no reason.
            img = img[..., 0]  # Just choosing first index value.

        img -= img[img > img.min()].min()
        img /= img.max()
        img[img < 0] = 0

        # fix the shape: (H, W, D) -> (D, C=1, H, W), the C setting as 1
        img = img.transpose((2, 0, 1))[:, np.newaxis, ...]
        img = np.ascontiguousarray(img)
        img = np.repeat(img, 3, axis=1)  # (D, C=3, H, W)
        meta = {'header': pack.header, 'idx': index % 256 if self.multi_gpu else index, 'fname': image_path}
        self.img_buf = [img, label, meta]
        return img, label, meta


class OneGPUDataset(Dataset):
    def __init__(self, label_path, image_path, config, target=LOC[-1], transform=default_T(), acc: Accelerator=None, multi_gpu=True):
        verify_image = pd.read_csv('./static/len_ct.csv')
        self.mapper = get_mapper(label_path)
        self.flist = verify_image['path'].tolist()
        self.except_sample = []
        
        if isinstance(transform, str):
            transform = eval(transform)
        self.transform = transform
        self.is_debug = config.get('debug', False)        

    def __len__(self):
        if self.is_debug:
            return 10
        return len(self.flist)

    def __getitem__(self, item):
        path = self.flist[item]
        pid = re.split('[/\\\]', path)[6]
        pack = nib.load(path)
        img = pack.get_fdata()
        if len(img.shape) > 3:
            img = np.ascontiguousarray(img[..., 0], dtype=np.float32)
        img -= img[img > -2000].min()
        img[img < 0] = 0
        img /= 1000.
        label = self.mapper[LOC[-1]][pid]
        img: torch.Tensor = self.transform(img).cpu().float()
        label = torch.as_tensor([label])
        # print(img.get_device(), label.get_device())                            
        return img, label, path


class MedicalReportDataset(Dataset):
    def __init__(self, csv_path, tokenizer):
        self.csv_path = csv_path
        self.tokenizer = tokenizer

        df = pd.read_excel(csv_path)


class ImageStack:
    def __init__(self, fully_ct, label, meta):
        self.ct_stack = np.split(fully_ct, fully_ct.shape[0])
        self.shape = self.ct_stack[0].shape
        self.min_intensity = np.min(fully_ct)
        self.label: float = label
        self.meta: dict = meta

    def pop(self, index):
        ct = self.ct_stack.pop(0) if len(self.ct_stack) > 0 else np.full(self.shape, self.min_intensity)
        label = self.label.copy()
        meta = self.meta.copy()
        meta['idx'] = index
        return ct, label, meta


def get_ds(config, acc: Accelerator=None):
    ds_config = config['dataset']
    ds_init = eval(ds_config['init'])

    return ds_init(**ds_config['init_param'], config=config, acc=acc)




if __name__ == '__main__':
    label_path = '/home/sdb_data/data/CT/CCTA_Agtston_Score.xlsx'
    image_path = '/home/sdb_data/data/CT/image'
    # ds = AgatstonScoreDataset(label_path, target=LOC[-1], image_path=image_path)
    ds = OneGPUDataset(label_path=label_path, image_path=image_path)
    loader = DataLoader(ds, collate_fn=collate_fn_no_meta, batch_size=1)
    from icecream import ic

    for i, pack in enumerate(loader):
        for keys, values in pack.items():
            try:
                ic(keys, type(values), values.size())
            except Exception as e:
                ic (keys, type(values))
        break