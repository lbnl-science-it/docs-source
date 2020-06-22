### [Jupyter Notebook on Lawrencium GPU node](https://github.com/lbnl-science-it/Lawrencium/blob/master/jupyter_gpu_tensorflow_2.1.ipynb)
* https://sites.google.com/a/lbl.gov/high-performance-computing-services-group/getting-started/jupyter-notebook

```shell
$ srun  -N 1 -p es1 -A $ACCOUNT -t 1:0:0 --gres=gpu:2 -n 4 -q es_normal --pty bash
$ module load ml/tensorflow/2.1.0-py37
$ start_jupyter.py
```
### Login from visualization node via VNC Viewer:
* https://sites.google.com/a/lbl.gov/high-performance-computing-services-group/getting-started/remote-desktop


```python
import sys
import numpy as np
import tensorflow as tf
from datetime import date
from datetime import datetime
```


```python
print(sys.version)
```

3.7.4 (default, Aug 13 2019, 20:35:49) 
[GCC 7.3.0]



```python
print(tf.compat.v1.VERSION)
```

2.1.0



```python
print(date.today())
```

2020-05-13


#### Test if TF can access a GPU


```python
tf.config.list_physical_devices('GPU')
```




    [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'),
     PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU'),
     PhysicalDevice(name='/physical_device:GPU:2', device_type='GPU'),
     PhysicalDevice(name='/physical_device:GPU:3', device_type='GPU')]



#### Print the name of the GPU device


```python
tf.test.gpu_device_name()
```

'/device:GPU:0'
