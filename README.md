# 1. Install environment

`conda env create -f environment.yaml`

`conda activate miod`

# 2. Download dataset

`python get_coco.py`

or

`python get_pascal.py`

# 3. Train

`python train.py`

# 4. Evaluate mAP

`python eval.py`
