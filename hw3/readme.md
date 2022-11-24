# CCBDA HW2

## Dataset

- Unzip mnist.zip to `./data/mnist`

    ```sh
    unzip mnist.zip -d ./data/mnist
    ```

- Folder structure

    ```txt
    ./
    ├── data
    │   └── mnist/
    │       ├── 00001.png
    │       ├── 00002.png
    │       ├── ...
    │       └── 60000.png
    ├── src/
    │   ├── __init__.py
    │   ├── dataset.py
    │   ├── diffusion.py
    │   ├── logger.py
    │   ├── models.py
    │   └── utils.py
    ├── config.yaml
    ├── readme.md
    ├── sample.py
    └── train.py
    ```

## Environment

- Python 3.10

### Create Environment

```sh
conda create --name hw3 python=3.10 --yes
conda activate hw3

conda install pytorch torchvision pytorch-cuda=11.6 -c pytorch -c nvidia --yes
conda install tqdm tensorboard pyyaml -c conda-forge --yes
```

### Activate Environment

```sh
conda activate hw3
```

## Train

```sh
python3 train.py [-h] [-c CONFIG] [-l LOGDIR]
```

- `-h` or `--help` : show this help message and exit
- `-c CONFIG` or `--config CONFIG` : config file path (default: `./config.yaml`)
- `-l LOGDIR` or `--logdir LOGDIR` : log dir name (default: current time)

### Config

- `config.yaml`

```yaml
train:
  epochs: 1000
  save_freq: 10
  batch_size: 128
  lr: !!float 1e-4
  ema_decay: !!float 0.9

model:
  time_steps: 1000
  channels: 64 # Must be divisible by 32, due to the group normalization.
  ch_mults: [1, 2, 2, 2]
  attn_res: [16, 8, 4] # Append self attention to the residual block at the specified resolution.
  num_res_blocks: 2
```

## Sample

```sh
python3 sample.py [-h] -l LOGDIR [-w WEIGHT]
```

- `-h` or `--help` : show this help message and exit
- `-l LOGDIR` or `--logdir LOGDIR` : log dir path, e.g. ./logs/2022-01-01_00-00-00
- `-w WEIGHT` or `--weight WEIGHT` : file name of model weight, e.g. ckpt_100 (default: latest)