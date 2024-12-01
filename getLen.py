import os
import nibabel as nib
import pandas as pd
from tqdm.auto import tqdm 
import re
df = pd.read_csv('./static/len_ct.csv')
already_exists = df['path'].to_list()
image_path = '/home/sdb_data/data/CT/image'
empty_file = []

len_df = {'path': [], 'z_size': []}
pbar = tqdm(os.walk(image_path, topdown=False), ncols=150, unit='file')

for roots, dirs, files in pbar:
    pid = re.split('[/\\\]', roots)[-1]
    pbar.set_postfix({'Root': roots})
    enable_files = list(filter(lambda x: x.endswith(".nii"), files))
    
    bar2 = tqdm(enable_files, ncols=150, total=len(enable_files), leave=False)
    bar2.set_description(f'Patient: {pid}')
    for file in bar2:
        if f'{roots}/{file}' in already_exists:
            bar2.set_postfix({'Error': 'Ignore', 'Shape': 'Ignore'})
            continue
        try:
            pack = nib.load(f'{roots}/{file}')
            data = pack.get_fdata()
        except Exception as e:
            empty_file.append(f'{roots}/{file}')
            bar2.set_postfix({'Error': str(e.args), 'Shape': None})
            continue
                
        len_df['path'].append(f'{roots}/{file}')
        len_df['z_size'].append(data.shape[2])
        bar2.set_postfix({'Shape': data.shape, 'Error': None})
        
# odf = df.to_dict(orient='list')
len_df = pd.DataFrame(len_df, columns=['path', 'z_size'])
len_df = pd.concat([df, len_df], axis=0, ignore_index=True).reset_index(drop=True)
len_df.to_csv('./static/len_ct.csv', index=False, index_label=False)

    

# import json

with open(f'./static/empty_nib.txt', 'w+') as fout:
    ebar = tqdm(empty_file, ncols=80, total=len(empty_file), desc=f'Store Empty File list|')
    for fpath in ebar:
        fout.write(f'{fpath}\n')
        ebar.update()