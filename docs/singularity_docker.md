## [Text Classification Using AWS Deep Learning Docker Containers](https://github.com/lbnl-science-it/singularity_aws_dl_container/blob/master/singularity_docker.ipynb)  

## Overview
In this tutorial, we will build a Singularity container using one of available [AWS Deep-Learning docker images](https://aws.amazon.com/releasenotes/available-deep-learning-containers-images/), and run the Singularity container on Lawrencium CPU and GPU nodes, to train an NLP model (based on Keras & TensorFlow). The NLP model will classify news articles into the appropriate news category given the training data from UCI News Dataset which contains a list of about 420K articles and their appropriate categories (labels). There are four categories:

* Business (b)
* Science & Technology (t)
* Entertainment (e)
* Health & Medicine (m)


## Prerequisites:
1. #### [AWS CLI version 1](https://docs.aws.amazon.com/cli/latest/userguide/install-linux.html)
1. #### [Docker](https://docs.docker.com/engine/install)
1. #### [Singularity](https://sylabs.io/guides/3.5/user-guide)


## Objectives
1) Build the Singularity container using available AWS Deep-Learning docker containers

2) Local Test

3) Train text classifier on Lawrencium:

   * Upload the Singularity containers and training data
   * Run the Singularity container on __CPU__ nodes
   * Run the Singularity container on __GPU__ nodes
   * CPU vs GPU


## Build the Singularity container using available AWS Deep-Learning docker containers 
* [AWS Deep-Learning images](https://aws.amazon.com/releasenotes/available-deep-learning-containers-images/)

The following shell code shows how to build the container image using `docker` and convert the container image to a `Singularity` image. 

### Download the GitHub repository for this tutorial

```sh
%%sh
git clone https://github.com/lbnl-science-it/singularity_aws_dl_container.git
cd singularity_aws_dl_container
```


### Download and unzip the dataset

```sh
%%sh
cd container

####################################################
########## Download and unzip the dataset ##########
####################################################
cd ../data/
wget https://danilop.s3-eu-west-1.amazonaws.com/reInvent-Workshop-Data-Backup.zip && unzip reInvent-Workshop-Data-Backup.zip
mv reInvent-Workshop-Data-Backup/* ./
rm -rf reInvent-Workshop-Data-Backup reInvent-Workshop-Data-Backup.zip
cd ../container/
```


### Build the SageMaker Container & Convert it to Singularity image

```sh
%%sh
cd container

###################################################################################
######### Build the SageMaker Container & Convert it to Singularity image #########
###################################################################################
algorithm_name=sagemaker-keras-text-classification

chmod +x sagemaker_keras_text_classification/train
chmod +x sagemaker_keras_text_classification/serve

## Get the region defined in the current configuration
region=$(aws configure get region)
fullname="local_${algorithm_name}:latest"

## Get the login command from ECR and execute it directly
$(aws ecr get-login --no-include-email --region ${region} --registry-ids 763104351884)

## Build the docker image locally with the image name
## In the "Dockerfile", modify the source image to select one of the available deep learning docker containers images:
## https://aws.amazon.com/releasenotes/available-deep-learning-containers-images
docker build  -t ${algorithm_name} .
docker tag ${algorithm_name} ${fullname}

## Build Singularity image from local docker image
sifname="local_sagemaker-keras-text-classification.sif"
sudo singularity build ${sifname} docker-daemon:${fullname}
```

    Login Succeeded
    Sending build context to Docker daemon  456.3MB
    Step 1/9 : FROM 763104351884.dkr.ecr.us-east-2.amazonaws.com/tensorflow-training:1.14.0-cpu-py36-ubuntu16.04
     ---> e6a210ff54e4
    Step 2/9 : RUN apt-get update &&     apt-get install -y nginx imagemagick graphviz
     ---> Using cache
     ---> 32ff2dce1af3
    Step 3/9 : RUN pip install --upgrade pip
     ---> Using cache
     ---> 4e1b65ea3a65
    Step 4/9 : RUN pip install gevent gunicorn flask tensorflow_hub seqeval graphviz nltk spacy tqdm
     ---> Using cache
     ---> d97c22f6de86
    Step 5/9 : RUN python -m spacy download en_core_web_sm
     ---> Using cache
     ---> 14c8854a1901
    Step 6/9 : RUN python -m spacy download en
     ---> Using cache
     ---> 185661d9e15d
    Step 7/9 : ENV PATH="/opt/program:${PATH}"
     ---> Using cache
     ---> b5d5c6867074
    Step 8/9 : COPY sagemaker_keras_text_classification /opt/program
     ---> Using cache
     ---> ac73b50bd646
    Step 9/9 : WORKDIR /opt/program
     ---> Using cache
     ---> c5fe52a83024
    Successfully built c5fe52a83024
    Successfully tagged sagemaker-keras-text-classification:latest

    
    [34mINFO:   [0m Starting build...
    Getting image source signatures
    Copying blob sha256:87e513ddb4a6ce37dabf3de74b0284d49e08f4d7a3f0de393e6a533577e00f11
    ...
    Copying config sha256:77b2a54a3da8891391f609455182127c0944edb40397fbaf24f9ec80a9be5460
    Writing manifest to image destination
    Storing signatures
    2020/06/08 22:14:47  info unpack layer: sha256:647dce8a9de5ada5719e82c2ff5408867fcaa83145665bea4103d3705c2326b1
    ...
    2020/06/08 22:14:49  info unpack layer: sha256:1df727cf7f1435f496890edded1650193af403065eff27929a8b374d5b36d743
    2020/06/08 22:14:49  info unpack layer: sha256:df2ccfca12a78a5c880fd30514c57c84f250a81c223915e124cff93833f6b5d2
    2020/06/08 22:14:49  info unpack layer: sha256:88f2c64e66817e60a415e82323d1a2d3f19ca75eb4ea9ae7692a2fccc09c2de5
    [34mINFO:   [0m Creating SIF file...
    [34mINFO:   [0m Build complete: local_sagemaker-keras-text-classification.sif


### Train Text Classifier


```sh
%%sh
cd container

################################
########## Local Test ########## 
################################
cd ../data
cp -a . ../container/local_test/test_dir/input/data/training/
cd ../container
cd local_test

### Train
sifname="local_sagemaker-keras-text-classification.sif"
./train_local.sh ../${sifname}
```

    Starting the training.
                                                   TITLE  ...      TIMESTAMP
    1  Fed official says weak data caused by weather,...  ...  1394470370698
    2  Fed's Charles Plosser sees high bar for change...  ...  1394470371207
    3  US open: Stocks fall after Fed official hints ...  ...  1394470371550
    4  Fed risks falling 'behind the curve', Charles ...  ...  1394470371793
    5  Fed's Plosser: Nasty Weather Has Curbed Job Gr...  ...  1394470372027
    
    [5 rows x 7 columns]
    Found 65990 unique tokens.
    Shape of data tensor: (422417, 100)
    Shape of label tensor: (422417, 4)
    x_train shape:  (337933, 100)
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding (Embedding)        (None, 100, 100)          1000000   
    _________________________________________________________________
    flatten (Flatten)            (None, 10000)             0         
    _________________________________________________________________
    dense (Dense)                (None, 2)                 20002     
    _________________________________________________________________
    dense_1 (Dense)              (None, 4)                 12        
    =================================================================
    Total params: 1,020,014
    Trainable params: 1,020,014
    Non-trainable params: 0
    _________________________________________________________________
    Train on 337933 samples, validate on 84484 samples
    Epoch 1/6
    337933/337933 - 25s - loss: 0.6788 - acc: 0.7409 - val_loss: 0.6146 - val_acc: 0.7757
    Epoch 2/6
    337933/337933 - 25s - loss: 0.5958 - acc: 0.7824 - val_loss: 0.5889 - val_acc: 0.7840
    Epoch 3/6
    337933/337933 - 25s - loss: 0.5778 - acc: 0.7882 - val_loss: 0.5755 - val_acc: 0.7893
    Epoch 4/6
    337933/337933 - 25s - loss: 0.5707 - acc: 0.7904 - val_loss: 0.5697 - val_acc: 0.7918
    Epoch 5/6
    337933/337933 - 25s - loss: 0.5673 - acc: 0.7918 - val_loss: 0.5684 - val_acc: 0.7915
    Epoch 6/6
    337933/337933 - 25s - loss: 0.5648 - acc: 0.7920 - val_loss: 0.5657 - val_acc: 0.7923
    Training complete. Now saving model to:  /opt/ml/model
    Test headline:  What Improved Tech Means for Electric, Self-Driving and Flying Cars
    Predicted category:  t


### Data Exploration


```python
import pandas as pd
import tensorflow as tf
import re
import numpy as np
import os

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils import to_categorical

column_names = ["TITLE", "URL", "PUBLISHER", "CATEGORY", "STORY", "HOSTNAME", "TIMESTAMP"]
news_dataset = pd.read_csv(os.path.join('./data', 'newsCorpora.csv'), names=column_names, header=None, delimiter='\t')
news_dataset.head()
```

| TITLE | URL | PUBLISHER | CATEGORY | STORY | HOSTNAME | TIMESTAMP |
| ---   | --- | ---       | ---      |  ---  |  ---     |  ---      |
|Fed official says weak data caused by weather,...|http://www.latimes.com/business/money/la-fi-mo...|Los Angeles Times|b|ddUyU0VZz0BRneMioxUPQVP6sIxvM|www.latimes.com|1394470370698|



```python
news_dataset.groupby(['CATEGORY']).size()
```

    CATEGORY
    b    115967
    e    152469
    m     45639
    t    108344
    dtype: int64


## Local Test
  * Build Singularity container using AWS Deep-Learning Docker __CPU__ image
```shell
  cd container
  sh build_singularity_local_test.sh
```

  * Build Singularity container using AWS Deep-Learning Docker __GPU__ image
```shell
  sh build_singularity_local_test_gpu.sh
```

## Train text classifier on Lawrencium
  * ### Upload the Singularity containers and training data
```shell
    sftp lrc-xfer.lbl.gov
    put local_sagemaker-keras-text-classification.sif
    put local_sagemaker-keras-text-classification_gpu.sif
    put -r local_test
```

  * ### Run the Singularity container on Lawrencium __CPU__ node
```shell
    ssh lrc-login.lbl.gov
    cd local_test
    srun  -N 1 -p lr4 -A $ACCOUNT -t 1:0:0 -q lr_normal --pty bash
    sh train_local.sh ../local_sagemaker-keras-text-classification.sif
```

  * ### Run the Singularity container on Lawrencium __GPU__ node
```shell
    ssh lrc-login.lbl.gov
    cd local_test
    srun  -N 1 -p es1 -A $ACCOUNT -t 1:0:0 --gres=gpu:2 -n 4 -q es_normal --pty bash
    sh train_local_gpu.sh ../local_sagemaker-keras-text-classification_gpu.sif
```

```shell
Starting the training.
                                               TITLE                                                URL          PUBLISHER CATEGORY                          STORY             HOSTNAME      TIMESTAMP
1  Fed official says weak data caused by weather,...  http://www.latimes.com/business/money/la-fi-mo...  Los Angeles Times        b  ddUyU0VZz0BRneMioxUPQVP6sIxvM      www.latimes.com  1394470370698
2  Fed's Charles Plosser sees high bar for change...  http://www.livemint.com/Politics/H2EvwJSK2VE6O...           Livemint        b  ddUyU0VZz0BRneMioxUPQVP6sIxvM     www.livemint.com  1394470371207
3  US open: Stocks fall after Fed official hints ...  http://www.ifamagazine.com/news/us-open-stocks...       IFA Magazine        b  ddUyU0VZz0BRneMioxUPQVP6sIxvM  www.ifamagazine.com  1394470371550
4  Fed risks falling 'behind the curve', Charles ...  http://www.ifamagazine.com/news/fed-risks-fall...       IFA Magazine        b  ddUyU0VZz0BRneMioxUPQVP6sIxvM  www.ifamagazine.com  1394470371793
5  Fed's Plosser: Nasty Weather Has Curbed Job Gr...  http://www.moneynews.com/Economy/federal-reser...          Moneynews        b  ddUyU0VZz0BRneMioxUPQVP6sIxvM    www.moneynews.com  1394470372027
Found 65990 unique tokens.
Shape of data tensor: (422417, 100)
Shape of label tensor: (422417, 4)
x_train shape:  (337933, 100)
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, 100, 100)          1000000   
_________________________________________________________________
flatten (Flatten)            (None, 10000)             0         
_________________________________________________________________
dense (Dense)                (None, 2)                 20002     
_________________________________________________________________
dense_1 (Dense)              (None, 4)                 12        
=================================================================
Total params: 1,020,014
Trainable params: 1,020,014
Non-trainable params: 0
_________________________________________________________________
2020-06-23 23:15:14.245462: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcuda.so.1
2020-06-23 23:15:16.127338: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x3672c00 executing computations on platform CUDA. Devices:
2020-06-23 23:15:16.127380: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): GeForce GTX 1080 Ti, Compute Capability 6.1
2020-06-23 23:15:16.127388: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (1): GeForce GTX 1080 Ti, Compute Capability 6.1
2020-06-23 23:15:16.127394: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (2): GeForce GTX 1080 Ti, Compute Capability 6.1
2020-06-23 23:15:16.127399: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (3): GeForce GTX 1080 Ti, Compute Capability 6.1
2020-06-23 23:15:16.154635: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2999685000 Hz
2020-06-23 23:15:16.154819: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x3685470 executing computations on platform Host. Devices:
2020-06-23 23:15:16.154871: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): <undefined>, <undefined>
2020-06-23 23:15:16.157342: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1640] Found device 0 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:02:00.0
2020-06-23 23:15:16.158085: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1640] Found device 1 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:03:00.0
2020-06-23 23:15:16.158826: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1640] Found device 2 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:81:00.0
2020-06-23 23:15:16.159564: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1640] Found device 3 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:82:00.0
2020-06-23 23:15:16.164279: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcudart.so.10.0
2020-06-23 23:15:16.245853: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcublas.so.10.0
2020-06-23 23:15:16.286560: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcufft.so.10.0
2020-06-23 23:15:16.311842: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcurand.so.10.0
2020-06-23 23:15:16.418419: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcusolver.so.10.0
2020-06-23 23:15:16.483660: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcusparse.so.10.0
2020-06-23 23:15:16.684841: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcudnn.so.7
2020-06-23 23:15:16.694707: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1763] Adding visible gpu devices: 0, 1, 2, 3
2020-06-23 23:15:16.697050: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcudart.so.10.0
2020-06-23 23:15:16.702301: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1181] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-06-23 23:15:16.702355: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1187]      0 1 2 3 
2020-06-23 23:15:16.702409: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1200] 0:   N Y N N 
2020-06-23 23:15:16.702424: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1200] 1:   Y N N N 
2020-06-23 23:15:16.702445: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1200] 2:   N N N Y 
2020-06-23 23:15:16.702465: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1200] 3:   N N Y N 
2020-06-23 23:15:16.711677: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1326] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10481 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
2020-06-23 23:15:16.713584: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1326] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:1 with 10481 MB memory) -> physical GPU (device: 1, name: GeForce GTX 1080 Ti, pci bus id: 0000:03:00.0, compute capability: 6.1)
2020-06-23 23:15:16.715242: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1326] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:2 with 10481 MB memory) -> physical GPU (device: 2, name: GeForce GTX 1080 Ti, pci bus id: 0000:81:00.0, compute capability: 6.1)
2020-06-23 23:15:16.716880: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1326] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:3 with 10481 MB memory) -> physical GPU (device: 3, name: GeForce GTX 1080 Ti, pci bus id: 0000:82:00.0, compute capability: 6.1)
Train on 337933 samples, validate on 84484 samples
Epoch 1/6
2020-06-23 23:15:21.391855: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcublas.so.10.0
337933/337933 - 18s - loss: 0.6149 - acc: 0.7661 - val_loss: 0.5618 - val_acc: 0.7933
Epoch 2/6
337933/337933 - 16s - loss: 0.5476 - acc: 0.7972 - val_loss: 0.5513 - val_acc: 0.7981
Epoch 3/6
337933/337933 - 16s - loss: 0.5391 - acc: 0.8008 - val_loss: 0.5402 - val_acc: 0.8021
Epoch 4/6
337933/337933 - 17s - loss: 0.5355 - acc: 0.8016 - val_loss: 0.5385 - val_acc: 0.8023
Epoch 5/6
337933/337933 - 17s - loss: 0.5333 - acc: 0.8029 - val_loss: 0.5374 - val_acc: 0.8013
Epoch 6/6
337933/337933 - 16s - loss: 0.5315 - acc: 0.8032 - val_loss: 0.5361 - val_acc: 0.8023
Training complete. Now saving model to:  /opt/ml/model

Test headline:  What Improved Tech Means for Electric, Self-Driving and Flying Cars
Predicted category:  t

```

* ### CPU vs GPU
In this tutorial we trained the same NLP model on Lawrencium nodes with and without GPU; the training took about __17s__ per epoch (337933 samples) on GPU node, and __23s__ per epoch on CPU node.

## References 
1. https://aws.amazon.com/releasenotes/available-deep-learning-containers-images
1. https://github.com/aws-samples/amazon-sagemaker-keras-text-classification
1. https://github.com/lbnl-science-it/aws-sagemaker-keras-text-classification
1. https://sylabs.io/guides/3.5/user-guide/
1. https://www.digitalocean.com/community/questions/how-to-fix-docker-got-permission-denied-while-trying-to-connect-to-the-docker-daemon-socket
1. https://docs.aws.amazon.com/cli/latest/userguide/install-linux.html