## Dataset
- Unzip data.zip to `./data`
    ```sh
    unzip data.zip -d ./data
    ```
- Folder structure
    ```
    .
    ├── data
    │   ├── test/
    │   └── train/
    ├── eval.py
    ├── Readme.md
    ├── requirements.txt
    └── train.py
    ```

## Environment
- Python 3.6 or later version
    ```sh
    pip install -r requirements.txt
    ```

## Train
```sh
python train.py
```

## Make Prediction
```sh
python eval.py
```
The prediction file is `prediction.csv`.
