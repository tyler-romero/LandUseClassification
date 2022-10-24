## Download a pretrained model

# We obtained a 50-layer ResNet pretrained on ImageNet from a link in the
# [TensorFlow models repo's slim subdirectory](https://github.com/tensorflow/models/tree/master/slim).
# The pretrained model can be obtained and unpacked with the code snippet below.
# Note that if you have not already done so, you will first need to
# [download or clone this repo](https://github.com/Azure/Embarrassingly-Parallel-Image-Classification),
# then update the variable name `repo_dir` below to point to the repo's root folder.

# Be sure to run file_table_creation.sql prior to downloading tf models.

import os
import tarfile
import urllib.request

import land_use.connection_settings as cs


urllib.request.urlretrieve('http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz',
                           os.path.join(cs.IMAGE_DIR, 'TFPretrainedModels', 'resnet_v1_50_2016_08_28.tar.gz'))
with tarfile.open(os.path.join(cs.IMAGE_DIR, 'TFPretrainedModels', 'resnet_v1_50_2016_08_28.tar.gz'), 'r:gz') as f:
    def is_within_directory(directory, target):
        
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
    
        prefix = os.path.commonprefix([abs_directory, abs_target])
        
        return prefix == abs_directory
    
    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not is_within_directory(path, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
    
        tar.extractall(path, members, numeric_owner=numeric_owner) 
        
    
    safe_extract(f, path=os.path.join(cs.IMAGE_DIR,"TFPretrainedModels"))
os.remove(os.path.join(cs.IMAGE_DIR, 'TFPretrainedModels',  'resnet_v1_50_2016_08_28.tar.gz'))