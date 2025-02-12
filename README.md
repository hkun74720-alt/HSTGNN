# HSTGNN
This is the pytorch implementation of HSTGNN. 

This paper has been accepted by [Information fusion 2025.](https://doi.org/10.1016/j.inffus.2025.102978)


## Dependencies
The code is built based on Python 3.8. Run the following code to install packages.

```
pip install tables pandas scipy Pywavelets
```

## Datasets
The datasets can be found in STSGCN [https://github.com/Davidham3/STSGCN] and DCRNN [https://github.com/liyaguang/DCRNN].

Run the following code to generate training data.

```
python data_pre.py
```
You can also download pprocessed data from [here](https://pan.baidu.com/s/1R7rc_NENS3AEoPGq3MCNCg?pwd=q5eu).
## Train Commands
Run the following code to train the model.

```
python train.py
```

We also provided pre-trained models, and you can run the following code to test the model.

```
python test.py
```
## Citation
If you find our work is helpful, please cite as:

