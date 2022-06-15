### Instruction

## Data preparation
1. Download MUSDB18 dataset and place it in data/
2. Run:
    ```
        python PreprocessorMusDB18.py
    ```
## How to inprint
1. Run:
```
    python inpaint.py -p val -c config/sr_musdb.json -gpu 0
```