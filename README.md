# HSTGNN
This is the pytorch implementation of HSTGNN. 

[This paper has been accepted by Information fusion 2025 now.](https://doi.org/10.1016/j.inffus.2025.102978)

<p align="center">
  <img src="figs/stamt_model.png" width="100%">
</p>

## Dependencies
The code is built based on Python 3.8, PyTorch 1.11.0, and Cuda 11.3.

## Datasets
The datasets can be found in STSGCN [https://github.com/Davidham3/STSGCN].

## Train Commands
Run the following code to train the model.

```
python train.py
```

We also provided pre-trained model, and yoou can run the following code to test the model.

```
python test.py
```

## Results
<p align="center">
<img src="figs/stamt_result1.png" width="100%">
</p>
<p align="center">
<img src="figs/stamt_result2.png" width="100%">
</p>
<p align="center">
<img src="figs/stamt_result3.png" width="55%">
</p>
