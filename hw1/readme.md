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

```bash
conda create --name ccbda-hw1 python=3.10
conda activate ccbda-hw1
conda install -c conda-forge ffmpeg

pip3 install -r requirements.txt
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
