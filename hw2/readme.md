# CCBDA HW2

## Dataset

- Unzip data.zip to `./data`

    ```sh
    unzip data.zip -d ./data
    ```

- Folder structure

    ```txt
    ./
    ├── data
    │   ├── test/
    │   └── unlabeled/
    ├── src/
    ├── config.yaml
    ├── readme.md
    ├── requirements.txt
    ├── embed.py
    └── train.py
    ```

## Environment

- Python 3.10

### Create Environment

```sh
conda create --name ccbda-hw2 python=3.10 --yes
conda activate ccbda-hw2

conda install pytorch torchvision cudatoolkit=11.6 -c pytorch -c conda-forge --yes
conda install numpy -c conda-forge --yes
conda install tqdm tensorboard pyyaml --yes
```

### Activate Environment

```sh
conda activate ccbda-hw2
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
  batch_size: 16
  update_batch_size: 1024
  epochs: 1000
  save_freq: 200

  lr: !!float 1e-4
  weight_decay: !!float 1e-6

model:
  encoder:
    # One of ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    name: resnet50
    latent_dim: 512
  projector:
    dims: [256, 128]
```

## Embedding

```sh
python3 embed.py [-h] -l LOGDIR [-w WEIGHT]
```

- `-h` or `--help` : show this help message and exit
- `-l LOGDIR` or `--logdir LOGDIR` : log dir path, e.g. ./logs/2022-01-01_00-00-00
- `-w WEIGHT` or `--weight WEIGHT` : file name of model weight, e.g. ckpt_100 (default: latest)
