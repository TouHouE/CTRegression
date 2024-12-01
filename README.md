## Prepare dataset & Configuration
- Modify the `config.json` under `./static`
## Training Regressor
### Multi-GPU training
```sh
export CUDA_VISIBLE_DEVICES=0,1,2...
accelerate launch main.py --gpu_ids="0,1,2..." --multi-gpu
```
