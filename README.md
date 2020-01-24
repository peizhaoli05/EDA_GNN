# Graph Neural Based End-to-end Data Association Framework for Online Multiple-Object Tracking
A PyTorch implementation combines with Siamese Network and Graph Neural Network for Online Multiple-Object Tracking.

Dataset available at [https://motchallenge.net/]

According paper can be found at [https://arxiv.org/abs/1907.05315]

## How to run
Use `python main.py` to train a model from scratch. Settings for training is in `config.yml`.  
Use `python tracking.py` to track a test video, meanwhile you need to provide the detected objects & tracking results for the first five frames. Setting for tracking is in `setting/`.

## Requirements
 - Python 2.7.12
 - numpy 1.11.0
 - scipy 1.1.0
 - torchvision 0.2.1
 - opencv_python 3.3.0.10
 - easydict 1.7
 - torch 0.4.1
 - Pillow 6.2.0
 - PyYAML 5.1
