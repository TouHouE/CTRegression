import matplotlib.pyplot as plt
import torch

for idx in range(4):
    plt.figure(figsize=(15, 15))
    tmp = torch.load(f'./static/cuda:{idx}_image_tensor.pth', map_location='cpu')[:, 0, ...]
    for z in range(tmp.shape[0]):
        plt.subplot(8, 8, z + 1)
        plt.imshow(tmp[z])
        plt.title(f'CUDA:{idx}-z: {z}')
        plt.axis('off')

    plt.savefig(f'./static/cuda:{idx}.png')
    plt.close()