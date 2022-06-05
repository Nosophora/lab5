```python
!pip install tflite-model-maker
```

    ^C
    


```python
!pip install conda-repo-cli==1.0.4
```

    Requirement already satisfied: conda-repo-cli==1.0.4 in d:\anaconda3\lib\site-packages (1.0.4)
    Requirement already satisfied: nbformat>=4.4.0 in d:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (5.1.3)
    Requirement already satisfied: six in d:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (1.16.0)
    Requirement already satisfied: requests>=2.9.1 in d:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (2.26.0)
    Requirement already satisfied: setuptools in d:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (58.0.4)
    Requirement already satisfied: pytz in d:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (2021.3)
    Requirement already satisfied: PyYAML>=3.12 in d:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (6.0)
    Requirement already satisfied: clyent>=1.2.0 in d:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (1.2.2)
    Collecting pathlib
      Downloading pathlib-1.0.1-py3-none-any.whl (14 kB)
    Requirement already satisfied: python-dateutil>=2.6.1 in d:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (2.8.2)
    Requirement already satisfied: jsonschema!=2.5.0,>=2.4 in d:\anaconda3\lib\site-packages (from nbformat>=4.4.0->conda-repo-cli==1.0.4) (3.2.0)
    Requirement already satisfied: traitlets>=4.1 in d:\anaconda3\lib\site-packages (from nbformat>=4.4.0->conda-repo-cli==1.0.4) (5.1.0)
    Requirement already satisfied: jupyter-core in d:\anaconda3\lib\site-packages (from nbformat>=4.4.0->conda-repo-cli==1.0.4) (4.8.1)
    Requirement already satisfied: ipython-genutils in d:\anaconda3\lib\site-packages (from nbformat>=4.4.0->conda-repo-cli==1.0.4) (0.2.0)
    Requirement already satisfied: pyrsistent>=0.14.0 in d:\anaconda3\lib\site-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.4.0->conda-repo-cli==1.0.4) (0.18.0)
    Requirement already satisfied: attrs>=17.4.0 in d:\anaconda3\lib\site-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.4.0->conda-repo-cli==1.0.4) (21.2.0)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in d:\anaconda3\lib\site-packages (from requests>=2.9.1->conda-repo-cli==1.0.4) (1.25.11)
    Requirement already satisfied: charset-normalizer~=2.0.0 in d:\anaconda3\lib\site-packages (from requests>=2.9.1->conda-repo-cli==1.0.4) (2.0.4)
    Requirement already satisfied: idna<4,>=2.5 in d:\anaconda3\lib\site-packages (from requests>=2.9.1->conda-repo-cli==1.0.4) (3.2)
    Requirement already satisfied: certifi>=2017.4.17 in d:\anaconda3\lib\site-packages (from requests>=2.9.1->conda-repo-cli==1.0.4) (2021.10.8)
    Requirement already satisfied: pywin32>=1.0 in d:\anaconda3\lib\site-packages (from jupyter-core->nbformat>=4.4.0->conda-repo-cli==1.0.4) (228)
    Installing collected packages: pathlib
    Successfully installed pathlib-1.0.1
    


```python
!pip install anaconda-project==0.10.1
```

    Requirement already satisfied: anaconda-project==0.10.1 in d:\anaconda3\lib\site-packages (0.10.1)
    Requirement already satisfied: requests in d:\anaconda3\lib\site-packages (from anaconda-project==0.10.1) (2.26.0)
    Requirement already satisfied: anaconda-client in d:\anaconda3\lib\site-packages (from anaconda-project==0.10.1) (1.9.0)
    Requirement already satisfied: jinja2 in d:\anaconda3\lib\site-packages (from anaconda-project==0.10.1) (2.11.3)
    Requirement already satisfied: tornado>=4.2 in d:\anaconda3\lib\site-packages (from anaconda-project==0.10.1) (6.1)
    Collecting ruamel-yaml
      Downloading ruamel.yaml-0.17.21-py3-none-any.whl (109 kB)
    Requirement already satisfied: conda-pack in d:\anaconda3\lib\site-packages (from anaconda-project==0.10.1) (0.6.0)
    Requirement already satisfied: clyent>=1.2.0 in d:\anaconda3\lib\site-packages (from anaconda-client->anaconda-project==0.10.1) (1.2.2)
    Requirement already satisfied: nbformat>=4.4.0 in d:\anaconda3\lib\site-packages (from anaconda-client->anaconda-project==0.10.1) (5.1.3)
    Requirement already satisfied: PyYAML>=3.12 in d:\anaconda3\lib\site-packages (from anaconda-client->anaconda-project==0.10.1) (6.0)
    Requirement already satisfied: setuptools in d:\anaconda3\lib\site-packages (from anaconda-client->anaconda-project==0.10.1) (58.0.4)
    Requirement already satisfied: six in d:\anaconda3\lib\site-packages (from anaconda-client->anaconda-project==0.10.1) (1.16.0)
    Requirement already satisfied: pytz in d:\anaconda3\lib\site-packages (from anaconda-client->anaconda-project==0.10.1) (2021.3)
    Requirement already satisfied: python-dateutil>=2.6.1 in d:\anaconda3\lib\site-packages (from anaconda-client->anaconda-project==0.10.1) (2.8.2)
    Requirement already satisfied: jupyter-core in d:\anaconda3\lib\site-packages (from nbformat>=4.4.0->anaconda-client->anaconda-project==0.10.1) (4.8.1)
    Requirement already satisfied: ipython-genutils in d:\anaconda3\lib\site-packages (from nbformat>=4.4.0->anaconda-client->anaconda-project==0.10.1) (0.2.0)
    Requirement already satisfied: traitlets>=4.1 in d:\anaconda3\lib\site-packages (from nbformat>=4.4.0->anaconda-client->anaconda-project==0.10.1) (5.1.0)
    Requirement already satisfied: jsonschema!=2.5.0,>=2.4 in d:\anaconda3\lib\site-packages (from nbformat>=4.4.0->anaconda-client->anaconda-project==0.10.1) (3.2.0)
    Requirement already satisfied: attrs>=17.4.0 in d:\anaconda3\lib\site-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.4.0->anaconda-client->anaconda-project==0.10.1) (21.2.0)
    Requirement already satisfied: pyrsistent>=0.14.0 in d:\anaconda3\lib\site-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.4.0->anaconda-client->anaconda-project==0.10.1) (0.18.0)
    Requirement already satisfied: idna<4,>=2.5 in d:\anaconda3\lib\site-packages (from requests->anaconda-project==0.10.1) (3.2)
    Requirement already satisfied: certifi>=2017.4.17 in d:\anaconda3\lib\site-packages (from requests->anaconda-project==0.10.1) (2021.10.8)
    Requirement already satisfied: charset-normalizer~=2.0.0 in d:\anaconda3\lib\site-packages (from requests->anaconda-project==0.10.1) (2.0.4)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in d:\anaconda3\lib\site-packages (from requests->anaconda-project==0.10.1) (1.25.11)
    Requirement already satisfied: MarkupSafe>=0.23 in d:\anaconda3\lib\site-packages (from jinja2->anaconda-project==0.10.1) (1.1.1)
    Requirement already satisfied: pywin32>=1.0 in d:\anaconda3\lib\site-packages (from jupyter-core->nbformat>=4.4.0->anaconda-client->anaconda-project==0.10.1) (228)
    Collecting ruamel.yaml.clib>=0.2.6
      Downloading ruamel.yaml.clib-0.2.6-cp39-cp39-win_amd64.whl (118 kB)
    Installing collected packages: ruamel.yaml.clib, ruamel-yaml
    Successfully installed ruamel-yaml-0.17.21 ruamel.yaml.clib-0.2.6
    


```python
import os

import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

import matplotlib.pyplot as plt

```


```python
!pip install tflite-model-maker
```

    Collecting tflite-model-maker
      Using cached tflite_model_maker-0.4.0-py3-none-any.whl (642 kB)
    Collecting absl-py>=0.10.0
      Using cached absl_py-1.0.0-py3-none-any.whl (126 kB)
    Collecting fire>=0.3.1
      Using cached fire-0.4.0-py2.py3-none-any.whl
    Collecting tensorflow-addons>=0.11.2
      Using cached tensorflow_addons-0.17.0-cp39-cp39-win_amd64.whl (758 kB)
    Collecting tflite-model-maker
      Using cached tflite_model_maker-0.3.4-py3-none-any.whl (616 kB)
    Collecting tensorflow>=2.6.0
      Using cached tensorflow-2.9.1-cp39-cp39-win_amd64.whl (444.0 MB)
    Collecting tensorflow-model-optimization>=0.5
      Using cached tensorflow_model_optimization-0.7.2-py2.py3-none-any.whl (237 kB)
    Collecting tflite-support>=0.3.1
      Using cached tflite_support-0.4.0-cp39-cp39-win_amd64.whl (439 kB)
    Collecting sentencepiece>=0.1.91
      Using cached sentencepiece-0.1.96-cp39-cp39-win_amd64.whl (1.1 MB)
    Requirement already satisfied: pillow>=7.0.0 in d:\anaconda3\lib\site-packages (from tflite-model-maker) (9.0.1)
    Collecting tensorflow-hub<0.13,>=0.7.0
      Using cached tensorflow_hub-0.12.0-py2.py3-none-any.whl (108 kB)
    Collecting numpy>=1.17.3
      Downloading numpy-1.22.4-cp39-cp39-win_amd64.whl (14.7 MB)
    Collecting flatbuffers==1.12
      Using cached flatbuffers-1.12-py2.py3-none-any.whl (15 kB)
    Collecting matplotlib<3.5.0,>=3.0.3
      Downloading matplotlib-3.4.3-cp39-cp39-win_amd64.whl (7.1 MB)
    Collecting lxml>=4.6.1
      Downloading lxml-4.8.0-cp39-cp39-win_amd64.whl (3.6 MB)
    Requirement already satisfied: PyYAML>=5.1 in d:\anaconda3\lib\site-packages (from tflite-model-maker) (6.0)
    Collecting neural-structured-learning>=1.3.1
      Using cached neural_structured_learning-1.3.1-py2.py3-none-any.whl (120 kB)
    Requirement already satisfied: six>=1.12.0 in d:\anaconda3\lib\site-packages (from tflite-model-maker) (1.16.0)
    Collecting tensorflow-datasets>=2.1.0
      Using cached tensorflow_datasets-4.5.2-py3-none-any.whl (4.2 MB)
    Collecting urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1
      Using cached urllib3-1.25.11-py2.py3-none-any.whl (127 kB)
    Collecting tensorflowjs>=2.4.0
      Using cached tensorflowjs-3.18.0-py3-none-any.whl (77 kB)
    Collecting Cython>=0.29.13
      Using cached Cython-0.29.30-py2.py3-none-any.whl (985 kB)
    Collecting librosa==0.8.1
      Using cached librosa-0.8.1-py3-none-any.whl (203 kB)
    Collecting numba==0.53
      Using cached numba-0.53.0-cp39-cp39-win_amd64.whl (2.3 MB)
    Collecting tf-models-official==2.3.0
      Using cached tf_models_official-2.3.0-py2.py3-none-any.whl (840 kB)
    Collecting scikit-learn!=0.19.0,>=0.14.0
      Downloading scikit_learn-1.1.1-cp39-cp39-win_amd64.whl (7.4 MB)
    Collecting resampy>=0.2.2
      Using cached resampy-0.2.2-py3-none-any.whl
    Collecting joblib>=0.14
      Using cached joblib-1.1.0-py2.py3-none-any.whl (306 kB)
    Requirement already satisfied: decorator>=3.0.0 in d:\anaconda3\lib\site-packages (from librosa==0.8.1->tflite-model-maker) (5.1.1)
    Collecting audioread>=2.0.0
      Using cached audioread-2.1.9-py3-none-any.whl
    Collecting pooch>=1.0
      Using cached pooch-1.6.0-py3-none-any.whl (56 kB)
    Requirement already satisfied: packaging>=20.0 in d:\anaconda3\lib\site-packages (from librosa==0.8.1->tflite-model-maker) (21.3)
    Collecting scipy>=1.0.0
      Downloading scipy-1.8.1-cp39-cp39-win_amd64.whl (36.9 MB)
    Collecting soundfile>=0.10.2
      Using cached SoundFile-0.10.3.post1-py2.py3.cp26.cp27.cp32.cp33.cp34.cp35.cp36.pp27.pp32.pp33-none-win_amd64.whl (689 kB)
    Collecting llvmlite<0.37,>=0.36.0rc1
      Using cached llvmlite-0.36.0-cp39-cp39-win_amd64.whl (16.0 MB)
    Requirement already satisfied: setuptools in d:\anaconda3\lib\site-packages (from numba==0.53->tflite-model-maker) (61.2.0)
    Collecting tf-slim>=1.1.0
      Using cached tf_slim-1.1.0-py2.py3-none-any.whl (352 kB)
    Collecting dataclasses
      Using cached dataclasses-0.6-py3-none-any.whl (14 kB)
    Collecting opencv-python-headless
      Using cached opencv_python_headless-4.5.5.64-cp36-abi3-win_amd64.whl (35.3 MB)
    Requirement already satisfied: psutil>=5.4.3 in d:\anaconda3\lib\site-packages (from tf-models-official==2.3.0->tflite-model-maker) (5.8.0)
    Collecting google-api-python-client>=1.6.7
      Using cached google_api_python_client-2.49.0-py2.py3-none-any.whl (8.5 MB)
    Collecting google-cloud-bigquery>=0.31.0
      Using cached google_cloud_bigquery-3.1.0-py2.py3-none-any.whl (211 kB)
    Collecting gin-config
      Using cached gin_config-0.5.0-py3-none-any.whl (61 kB)
    Collecting kaggle>=1.3.9
      Using cached kaggle-1.5.12-py3-none-any.whl
    Collecting pandas>=0.22.0
      Downloading pandas-1.4.2-cp39-cp39-win_amd64.whl (10.5 MB)
    Collecting py-cpuinfo>=3.3.0
      Using cached py_cpuinfo-8.0.0-py3-none-any.whl
    Collecting termcolor
      Using cached termcolor-1.1.0-py3-none-any.whl
    Collecting google-auth-httplib2>=0.1.0
      Using cached google_auth_httplib2-0.1.0-py2.py3-none-any.whl (9.3 kB)
    Collecting uritemplate<5,>=3.0.1
      Using cached uritemplate-4.1.1-py2.py3-none-any.whl (10 kB)
    Requirement already satisfied: google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.0,<3.0.0dev,>=1.31.5 in d:\anaconda3\lib\site-packages (from google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (2.8.0)
    Requirement already satisfied: google-auth<3.0.0dev,>=1.16.0 in d:\anaconda3\lib\site-packages (from google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (2.6.6)
    Collecting httplib2<1dev,>=0.15.0
      Using cached httplib2-0.20.4-py3-none-any.whl (96 kB)
    Requirement already satisfied: protobuf>=3.12.0 in d:\anaconda3\lib\site-packages (from google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.0,<3.0.0dev,>=1.31.5->google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (3.19.4)
    Requirement already satisfied: requests<3.0.0dev,>=2.18.0 in d:\anaconda3\lib\site-packages (from google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.0,<3.0.0dev,>=1.31.5->google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (2.27.1)
    Requirement already satisfied: googleapis-common-protos<2.0dev,>=1.52.0 in d:\anaconda3\lib\site-packages (from google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.0,<3.0.0dev,>=1.31.5->google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (1.56.1)
    Requirement already satisfied: rsa<5,>=3.1.4 in d:\anaconda3\lib\site-packages (from google-auth<3.0.0dev,>=1.16.0->google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (4.8)
    Requirement already satisfied: cachetools<6.0,>=2.0.0 in d:\anaconda3\lib\site-packages (from google-auth<3.0.0dev,>=1.16.0->google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (5.1.0)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in d:\anaconda3\lib\site-packages (from google-auth<3.0.0dev,>=1.16.0->google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (0.2.8)
    Collecting pyarrow<9.0dev,>=3.0.0
      Using cached pyarrow-8.0.0-cp39-cp39-win_amd64.whl (17.9 MB)
    Requirement already satisfied: proto-plus>=1.15.0 in d:\anaconda3\lib\site-packages (from google-cloud-bigquery>=0.31.0->tf-models-official==2.3.0->tflite-model-maker) (1.20.4)
    Collecting google-resumable-media<3.0dev,>=0.6.0
      Using cached google_resumable_media-2.3.3-py2.py3-none-any.whl (76 kB)
    Requirement already satisfied: python-dateutil<3.0dev,>=2.7.2 in d:\anaconda3\lib\site-packages (from google-cloud-bigquery>=0.31.0->tf-models-official==2.3.0->tflite-model-maker) (2.8.2)
    Collecting google-cloud-core<3.0.0dev,>=1.4.1
      Using cached google_cloud_core-2.3.0-py2.py3-none-any.whl (29 kB)
    Collecting google-cloud-bigquery-storage<3.0.0dev,>=2.0.0
      Using cached google_cloud_bigquery_storage-2.13.1-py2.py3-none-any.whl (180 kB)
    Requirement already satisfied: grpcio<2.0dev,>=1.38.1 in d:\anaconda3\lib\site-packages (from google-cloud-bigquery>=0.31.0->tf-models-official==2.3.0->tflite-model-maker) (1.46.3)
    Requirement already satisfied: grpcio-status<2.0dev,>=1.33.2 in d:\anaconda3\lib\site-packages (from google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.0,<3.0.0dev,>=1.31.5->google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (1.46.3)
    Collecting google-crc32c<2.0dev,>=1.0
      Using cached google_crc32c-1.3.0-cp39-cp39-win_amd64.whl (27 kB)
    Requirement already satisfied: pyparsing!=3.0.0,!=3.0.1,!=3.0.2,!=3.0.3,<4,>=2.4.2 in d:\anaconda3\lib\site-packages (from httplib2<1dev,>=0.15.0->google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (3.0.4)
    Requirement already satisfied: certifi in d:\anaconda3\lib\site-packages (from kaggle>=1.3.9->tf-models-official==2.3.0->tflite-model-maker) (2022.5.18.1)
    Collecting python-slugify
      Using cached python_slugify-6.1.2-py2.py3-none-any.whl (9.4 kB)
    Requirement already satisfied: tqdm in d:\anaconda3\lib\site-packages (from kaggle>=1.3.9->tf-models-official==2.3.0->tflite-model-maker) (4.64.0)
    Collecting kiwisolver>=1.0.1
      Downloading kiwisolver-1.4.2-cp39-cp39-win_amd64.whl (55 kB)
    Collecting cycler>=0.10
      Using cached cycler-0.11.0-py3-none-any.whl (6.4 kB)
    Requirement already satisfied: attrs in d:\anaconda3\lib\site-packages (from neural-structured-learning>=1.3.1->tflite-model-maker) (21.4.0)
    Requirement already satisfied: pytz>=2020.1 in d:\anaconda3\lib\site-packages (from pandas>=0.22.0->tf-models-official==2.3.0->tflite-model-maker) (2021.3)
    Collecting appdirs>=1.3.0
      Using cached appdirs-1.4.4-py2.py3-none-any.whl (9.6 kB)
    Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in d:\anaconda3\lib\site-packages (from pyasn1-modules>=0.2.1->google-auth<3.0.0dev,>=1.16.0->google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (0.4.8)
    Requirement already satisfied: idna<4,>=2.5 in d:\anaconda3\lib\site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.0,<3.0.0dev,>=1.31.5->google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (3.3)
    Requirement already satisfied: charset-normalizer~=2.0.0 in d:\anaconda3\lib\site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.0,<3.0.0dev,>=1.31.5->google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (2.0.4)
    Collecting threadpoolctl>=2.0.0
      Using cached threadpoolctl-3.1.0-py3-none-any.whl (14 kB)
    Requirement already satisfied: cffi>=1.0 in d:\anaconda3\lib\site-packages (from soundfile>=0.10.2->librosa==0.8.1->tflite-model-maker) (1.15.0)
    Requirement already satisfied: pycparser in d:\anaconda3\lib\site-packages (from cffi>=1.0->soundfile>=0.10.2->librosa==0.8.1->tflite-model-maker) (2.21)
    Collecting gast<=0.4.0,>=0.2.1
      Using cached gast-0.4.0-py3-none-any.whl (9.8 kB)
    Collecting tensorflow-io-gcs-filesystem>=0.23.1
      Using cached tensorflow_io_gcs_filesystem-0.26.0-cp39-cp39-win_amd64.whl (1.5 MB)
    Collecting keras<2.10.0,>=2.9.0rc0
      Using cached keras-2.9.0-py2.py3-none-any.whl (1.6 MB)
    Collecting keras-preprocessing>=1.1.1
      Using cached Keras_Preprocessing-1.1.2-py2.py3-none-any.whl (42 kB)
    Collecting h5py>=2.9.0
      Downloading h5py-3.7.0-cp39-cp39-win_amd64.whl (2.6 MB)
    Collecting opt-einsum>=2.3.2
      Using cached opt_einsum-3.3.0-py3-none-any.whl (65 kB)
    Collecting libclang>=13.0.0
      Using cached libclang-14.0.1-py2.py3-none-win_amd64.whl (14.2 MB)
    Requirement already satisfied: typing-extensions>=3.6.6 in d:\anaconda3\lib\site-packages (from tensorflow>=2.6.0->tflite-model-maker) (4.1.1)
    Collecting wrapt>=1.11.0
      Downloading wrapt-1.14.1-cp39-cp39-win_amd64.whl (35 kB)
    Collecting google-pasta>=0.1.1
      Using cached google_pasta-0.2.0-py3-none-any.whl (57 kB)
    Collecting tensorflow-estimator<2.10.0,>=2.9.0rc0
      Using cached tensorflow_estimator-2.9.0-py2.py3-none-any.whl (438 kB)
    Collecting astunparse>=1.6.0
      Using cached astunparse-1.6.3-py2.py3-none-any.whl (12 kB)
    Collecting tensorboard<2.10,>=2.9
      Using cached tensorboard-2.9.0-py3-none-any.whl (5.8 MB)
    Requirement already satisfied: wheel<1.0,>=0.23.0 in d:\anaconda3\lib\site-packages (from astunparse>=1.6.0->tensorflow>=2.6.0->tflite-model-maker) (0.37.1)
    Collecting werkzeug>=1.0.1
      Using cached Werkzeug-2.1.2-py3-none-any.whl (224 kB)
    Requirement already satisfied: tensorboard-data-server<0.7.0,>=0.6.0 in d:\anaconda3\lib\site-packages (from tensorboard<2.10,>=2.9->tensorflow>=2.6.0->tflite-model-maker) (0.6.1)
    Requirement already satisfied: markdown>=2.6.8 in d:\anaconda3\lib\site-packages (from tensorboard<2.10,>=2.9->tensorflow>=2.6.0->tflite-model-maker) (3.3.7)
    Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in d:\anaconda3\lib\site-packages (from tensorboard<2.10,>=2.9->tensorflow>=2.6.0->tflite-model-maker) (1.8.1)
    Collecting google-auth-oauthlib<0.5,>=0.4.1
      Using cached google_auth_oauthlib-0.4.6-py2.py3-none-any.whl (18 kB)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in d:\anaconda3\lib\site-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.10,>=2.9->tensorflow>=2.6.0->tflite-model-maker) (1.3.1)
    Collecting importlib-metadata>=4.4
      Using cached importlib_metadata-4.11.4-py3-none-any.whl (18 kB)
    Collecting zipp>=0.5
      Using cached zipp-3.8.0-py3-none-any.whl (5.4 kB)
    Requirement already satisfied: oauthlib>=3.0.0 in d:\anaconda3\lib\site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.10,>=2.9->tensorflow>=2.6.0->tflite-model-maker) (3.2.0)
    Collecting typeguard>=2.7
    

    WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ReadTimeoutError("HTTPSConnectionPool(host='pypi.org', port=443): Read timed out. (read timeout=15)")': /simple/py-cpuinfo/
    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    jupyter-server 1.13.5 requires pywinpty<2; os_name == "nt", but you have pywinpty 2.0.2 which is incompatible.
    

      Using cached typeguard-2.13.3-py3-none-any.whl (17 kB)
    Collecting tensorflow-metadata
      Using cached tensorflow_metadata-1.8.0-py3-none-any.whl (50 kB)
    Collecting dill
      Using cached dill-0.3.5.1-py2.py3-none-any.whl (95 kB)
    Collecting promise
      Using cached promise-2.3-py3-none-any.whl
    Collecting dm-tree~=0.1.1
      Using cached dm_tree-0.1.7-cp39-cp39-win_amd64.whl (90 kB)
    Collecting packaging>=20.0
      Using cached packaging-20.9-py2.py3-none-any.whl (40 kB)
    Collecting sounddevice>=0.4.4
      Using cached sounddevice-0.4.4-py3-none-win_amd64.whl (195 kB)
    Collecting pybind11>=2.6.0
      Using cached pybind11-2.9.2-py2.py3-none-any.whl (213 kB)
    Collecting text-unidecode>=1.3
      Using cached text_unidecode-1.3-py2.py3-none-any.whl (78 kB)
    Requirement already satisfied: colorama in d:\anaconda3\lib\site-packages (from tqdm->kaggle>=1.3.9->tf-models-official==2.3.0->tflite-model-maker) (0.4.4)
    Installing collected packages: urllib3, zipp, importlib-metadata, werkzeug, text-unidecode, numpy, llvmlite, httplib2, google-crc32c, google-auth-oauthlib, absl-py, wrapt, uritemplate, typeguard, threadpoolctl, termcolor, tensorflow-metadata, tensorflow-io-gcs-filesystem, tensorflow-estimator, tensorboard, scipy, python-slugify, pyarrow, promise, packaging, opt-einsum, numba, libclang, kiwisolver, keras-preprocessing, keras, joblib, h5py, google-resumable-media, google-pasta, google-cloud-core, google-cloud-bigquery-storage, google-auth-httplib2, gast, flatbuffers, dm-tree, dill, cycler, astunparse, appdirs, tf-slim, tensorflow-model-optimization, tensorflow-hub, tensorflow-datasets, tensorflow-addons, tensorflow, soundfile, sounddevice, sentencepiece, scikit-learn, resampy, pybind11, py-cpuinfo, pooch, pandas, opencv-python-headless, matplotlib, kaggle, google-cloud-bigquery, google-api-python-client, gin-config, dataclasses, Cython, audioread, tflite-support, tf-models-official, tensorflowjs, neural-structured-learning, lxml, librosa, fire, tflite-model-maker
      Attempting uninstall: urllib3
        Found existing installation: urllib3 1.26.9
        Uninstalling urllib3-1.26.9:
          Successfully uninstalled urllib3-1.26.9
      Attempting uninstall: packaging
        Found existing installation: packaging 21.3
        Uninstalling packaging-21.3:
          Successfully uninstalled packaging-21.3
    Successfully installed Cython-0.29.30 absl-py-1.0.0 appdirs-1.4.4 astunparse-1.6.3 audioread-2.1.9 cycler-0.11.0 dataclasses-0.6 dill-0.3.5.1 dm-tree-0.1.7 fire-0.4.0 flatbuffers-1.12 gast-0.4.0 gin-config-0.5.0 google-api-python-client-2.49.0 google-auth-httplib2-0.1.0 google-auth-oauthlib-0.4.6 google-cloud-bigquery-3.1.0 google-cloud-bigquery-storage-2.13.1 google-cloud-core-2.3.0 google-crc32c-1.3.0 google-pasta-0.2.0 google-resumable-media-2.3.3 h5py-3.7.0 httplib2-0.20.4 importlib-metadata-4.11.4 joblib-1.1.0 kaggle-1.5.12 keras-2.9.0 keras-preprocessing-1.1.2 kiwisolver-1.4.2 libclang-14.0.1 librosa-0.8.1 llvmlite-0.36.0 lxml-4.8.0 matplotlib-3.4.3 neural-structured-learning-1.3.1 numba-0.53.0 numpy-1.22.4 opencv-python-headless-4.5.5.64 opt-einsum-3.3.0 packaging-20.9 pandas-1.4.2 pooch-1.6.0 promise-2.3 py-cpuinfo-8.0.0 pyarrow-8.0.0 pybind11-2.9.2 python-slugify-6.1.2 resampy-0.2.2 scikit-learn-1.1.1 scipy-1.8.1 sentencepiece-0.1.96 sounddevice-0.4.4 soundfile-0.10.3.post1 tensorboard-2.9.0 tensorflow-2.9.1 tensorflow-addons-0.17.0 tensorflow-datasets-4.5.2 tensorflow-estimator-2.9.0 tensorflow-hub-0.12.0 tensorflow-io-gcs-filesystem-0.26.0 tensorflow-metadata-1.8.0 tensorflow-model-optimization-0.7.2 tensorflowjs-3.18.0 termcolor-1.1.0 text-unidecode-1.3 tf-models-official-2.3.0 tf-slim-1.1.0 tflite-model-maker-0.3.4 tflite-support-0.4.0 threadpoolctl-3.1.0 typeguard-2.13.3 uritemplate-4.1.1 urllib3-1.25.11 werkzeug-2.1.2 wrapt-1.14.1 zipp-3.8.0
    


```python
image_path = tf.keras.utils.get_file(
      'flower_photos.tgz',
      'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
      extract=True)
image_path = os.path.join(os.path.dirname(image_path), 'flower_photos')

```


```python
data = DataLoader.from_folder(image_path)
train_data, test_data = data.split(0.9)
```

    INFO:tensorflow:Load image with size: 3670, num_label: 5, labels: daisy, dandelion, roses, sunflowers, tulips.
    


```python
inception_v3_spec = image_classifier.ModelSpec(uri='https://storage.googleapis.com/tfhub-modules/tensorflow/efficientnet/lite0/feature-vector/2.tar.gz')
inception_v3_spec.input_image_shape = [240, 240]
model = image_classifier.create(train_data, model_spec=inception_v3_spec)
```

    INFO:tensorflow:Retraining the models...
    

    INFO:tensorflow:Retraining the models...
    

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     hub_keras_layer_v1v2 (HubKe  (None, 1280)             3413024   
     rasLayerV1V2)                                                   
                                                                     
     dropout (Dropout)           (None, 1280)              0         
                                                                     
     dense (Dense)               (None, 5)                 6405      
                                                                     
    =================================================================
    Total params: 3,419,429
    Trainable params: 6,405
    Non-trainable params: 3,413,024
    _________________________________________________________________
    None
    Epoch 1/5
    

    D:\anaconda3\lib\site-packages\keras\optimizers\optimizer_v2\gradient_descent.py:108: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
      super(SGD, self).__init__(name, **kwargs)
    

    103/103 [==============================] - 114s 1s/step - loss: 0.8645 - accuracy: 0.7831
    Epoch 2/5
    103/103 [==============================] - 89s 859ms/step - loss: 0.6513 - accuracy: 0.8999
    Epoch 3/5
    103/103 [==============================] - 100s 968ms/step - loss: 0.6205 - accuracy: 0.9187
    Epoch 4/5
    103/103 [==============================] - 102s 992ms/step - loss: 0.6009 - accuracy: 0.9287
    Epoch 5/5
    103/103 [==============================] - 89s 866ms/step - loss: 0.5906 - accuracy: 0.9375
    


```python
loss, accuracy = model.evaluate(test_data)
```

    12/12 [==============================] - 17s 1s/step - loss: 0.6603 - accuracy: 0.8910
    


```python
model.export(export_dir='.')
```

    INFO:tensorflow:Assets written to: C:\Users\asus\AppData\Local\Temp\tmp055txsta\assets
    

    INFO:tensorflow:Assets written to: C:\Users\asus\AppData\Local\Temp\tmp055txsta\assets
    D:\anaconda3\lib\site-packages\tensorflow\lite\python\convert.py:766: UserWarning: Statistics for quantized inputs were expected, but not specified; continuing anyway.
      warnings.warn("Statistics for quantized inputs were expected, but not "
    


```python

```
