# CCBDA-HW1

## Dataset

- Unzip data.zip to `./data`

    ```sh
    unzip data.zip -d ./data
    ```

- Folder structure

    ```txt
    .
    ├── data
    │   ├── test/
    │   └── train/
    ├── eval.py
    ├── readme.md
    ├── requirements.txt
    └── train.py
    ```

## Environment

### Create Environment

```sh
sudo apt install ffmpeg

conda create --name ccbda-hw1 python=3.10 --yes
conda activate ccbda-hw1
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 --yes

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

pip3 install -r requirements.txt
```

### Activate Environment

```sh
conda activate ccbda-hw1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
```

or

```sh
source activate.sh
```

## Train

```sh
python3 train.py
```

## Make Prediction

```sh
python3 eval.py
```

The prediction file is `prediction.csv`.
