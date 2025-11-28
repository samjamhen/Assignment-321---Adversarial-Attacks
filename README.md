# 321 Assignment

**NOTE:** Data folder is not pushed.

## Steps to Run:
1. Create a virtual environment and install required dependencies found in requirements.txt with pip install -r requirements.txt
2. Run `setup_dataset/download_UNSW_NB15.py` to download the dataset.
3. Take the output path and add it to `move_dataset.py`.
4. Run `move_dataset.py`.
5. Run `data_prep.py`.
6. Run `validate_data_prep.py`.

## For the Attack:
1. Run `train_models.py` to train the model.
2. Run `transfer_attack_art.py` to perform the attack.
