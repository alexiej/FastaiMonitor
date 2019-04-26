# FastaiMonitor
Simple two Monitors for fast.ai, using TensorBoard or Neptune.ml

`Colab Example:` https://colab.research.google.com/drive/1HEVbnOyKSLSIkMhpYLWO83Wut3hJrP42

# How to use with TensorBord

0. If you want to use Google Collab just 

1. install tensorboardX, and clone repository

```
pip install tensorboardX
git clone https://github.com/alexiej/FastaiMonitor.git
```

2. run tensorboard

`tensorboard --logdir ./log --host 0.0.0.0 --port 6006 &`

3. Add monitor for your learning fast.ai model


```python
from fastai import *
from fastai.vision import *
from FastaiMonitor.TensorBoardMonitor import TensorBoardMonitor

path = untar_data(URLs.MNIST_TINY);
bs = 64
data = ImageDataBunch.from_folder(path,
                                   ds_tfms=get_transforms(), 
                                   size=224, 
                                   bs=bs
                                  ).normalize(imagenet_stats)
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
TensorBoardMonitor(learn)

learn.fit_one_cycle(2, max_lr=1e-2)
```

4. Go to the tensorboard 





