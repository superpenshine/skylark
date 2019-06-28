# skylark
Astro project, interframe prediction of protoplanetary disk using resnet.

<img src="https://github.com/superpenshine/skylark/blob/master/img/example.jpg" alt="Protoplanetary disk" width="320">

## Requirements

cuda9.0.176
cudnn7.1/7.5

Python requirements:

* pytorch
* numpy
* matplotlib
* h5py
* tensorboardX
* Future
* Pillow
* opencv-python
* scipy


## Usage
- Save raw data to h5py files

```python main.py -m save```

- Train model

```python main.py```

- Visualize

```python main.py -m v```

- Clean up file/directory from last run

```python main.py -m c```
