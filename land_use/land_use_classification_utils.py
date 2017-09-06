import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from scipy.ndimage import imread
import functools
import tensorflow as tf
from .resnet_v1 import resnet_arg_scope, resnet_v1_50
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn
slim = tf.contrib.slim

import revoscalepy as revo
import microsoftml as ml

import land_use.connection_settings as cs


def gather_image_paths(images_folder, label_to_number_dict, connection_string, mode="train"):
    table = None
    if mode == "train":
        table = "TrainData"
    elif mode == "val":
        table = "ValData"
    elif mode == "test":
        table = "TestData"
    else:
        raise ValueError("'mode' not recognized")

    query = 'SELECT ([file_stream].GetFileNamespacePath()) as image FROM [{}] WHERE [is_directory] = 0 AND file_type = \'png\''.format(table)
    filetable_sql = revo.RxSqlServerData(sql_query=query, connection_string=connection_string)
    data = revo.rx_import(filetable_sql)

    data["image"] = data["image"].apply(lambda x: os.path.join(images_folder, x[1:]))    # TODO: assert to confirm paths exist
    data["class"] = data["image"].map(lambda x: os.path.basename(os.path.dirname(x)))
    data["label"] = data["class"].map(lambda x: label_to_number_dict[x])
    print(data.iloc[0,0])
    return data


def gather_image_paths2(data, label_to_number_dict):
    data["class"] = data["image"].map(lambda x: os.path.basename(os.path.dirname(x)))
    data["label"] = data["class"].map(lambda x: label_to_number_dict[x])
    print(data.iloc[0,0])
    return data


def featurize_transform(dataset, context):
    from microsoftml import load_image, resize_image, extract_pixels, featurize_image, rx_featurize
    from lung_cancer.connection_settings import MICROSOFTML_MODEL_NAME
    data = DataFrame(dataset)
    data = rx_featurize(
        data=data,
        overwrite=True,
        ml_transforms=[
            load_image(cols={"feature": "image"}),
            resize_image(cols="feature", width=224, height=224),
            extract_pixels(cols="feature"),
            featurize_image(cols="feature", dnn_model=MICROSOFTML_MODEL_NAME)
        ],
        ml_transform_vars=["image", "label"]
    )
    data.columns = ["image", "class", "label"] + ["f" + str(i) for i in range(len(data.columns)-3)]
    return data


def compute_features(data, output_table, connection_string):
    results_sql = revo.RxSqlServerData(table=output_table, connection_string=connection_string)
    revo.rx_data_step(input_data=data, output_file=results_sql, overwrite=True, transform_function=featurize_transform)


def create_formula(data, is_dataframe=False):
    if is_dataframe:
        features_all = list(data.columns)
    else:
        features_all = revo.rx_get_var_names(data)
    features_to_remove = ["label", "image", "class"]
    training_features = [x for x in features_all if x not in features_to_remove]
    formula = "label ~ " + " + ".join(training_features)
    return formula


def insert_model(table_name, connection_string, classifier, name):
    classifier_odbc = revo.RxOdbcData(connection_string, table=table_name)
    revo.rx_write_object(classifier_odbc, key=name, value=classifier, serialize=True, overwrite=True)


def retrieve_model(table_name, connection_string, name):
    classifier_odbc = revo.RxOdbcData(connection_string, table=table_name)
    classifier = revo.rx_read_object(classifier_odbc, key=name, deserialize=True)
    return classifier


# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


#-------------------------------------
# Original Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modified 2017 by Microsoft Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
class ImageReader(object):
    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_png_data = tf.placeholder(dtype=tf.string)
        self._decode_png = tf.image.decode_png(self._decode_png_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_png(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_png(self, sess, image_data):
        image = sess.run(self._decode_png,
                         feed_dict={self._decode_png_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image
    
def image_to_tfexample(image_data, image_format, height, width, class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format])),
        'image/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=[class_id])),
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
    }))

def find_images(image_dir):
    training_filenames = []
    for folder in os.listdir(image_dir):
        folder_path = os.path.join(image_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        ''' This is a new directory/label -- consider all images inside it '''
        my_filenames = []
        for filename in os.listdir(folder_path):
            my_filenames.append(os.path.join(folder_path, filename))
        training_filenames.extend(my_filenames)
    print('Found {} training images'.format(len(training_filenames)))
    return(training_filenames)
    
def write_dataset(dataset_name, split_name, my_filenames,  image_dir, n_shards=5):
    num_per_shard = int(np.ceil(len(my_filenames) / n_shards))
    records = []
    with tf.Graph().as_default():
        image_reader = ImageReader()
        with tf.Session('') as sess:
            for shard_idx in range(n_shards):
                shard_filename = os.path.join(image_dir,
                                              '{}_{}_{:05d}-of-{:05d}.tfrecord'.format(dataset_name,
                                                                                       split_name,
                                                                                       shard_idx+1,
                                                                                       n_shards))
                with tf.python_io.TFRecordWriter(shard_filename) as tfrecord_writer:
                    for image_idx in range(num_per_shard * shard_idx,
                                           min(num_per_shard * (shard_idx+1), len(my_filenames))):
                        with open(my_filenames[image_idx], 'rb') as f:
                            image_data = f.read()
                        height, width = image_reader.read_image_dims(sess, image_data)
                        class_name = os.path.basename(os.path.dirname(my_filenames[image_idx]))
                        class_id = cs.LABELS[class_name]
                        example = image_to_tfexample(image_data, b'png', height, width, class_id)
                        tfrecord_writer.write(example.SerializeToString())
                        records.append([dataset_name, split_name, my_filenames[image_idx], shard_idx,
                                        image_idx, class_name, class_id])
    df = pd.DataFrame(records, columns=['dataset_name', 'split_name', 'filename', 'shard_idx', 'image_idx',
                                        'class_name', 'class_id'])
    return(df)

def get_network_fn(num_classes, weight_decay=0.0, is_training=False):
    arg_scope = resnet_arg_scope(weight_decay=weight_decay)
    func = resnet_v1_50
    @functools.wraps(func)
    def network_fn(images):
        with slim.arg_scope(arg_scope):
            return func(images, num_classes, is_training=is_training)
    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size
    return(network_fn)

def mean_image_subtraction(image, means):
    num_channels = image.get_shape().as_list()[-1]
    channels = tf.split(image, num_channels, 2)
    for i in range(num_channels):
        channels[i] -= means[i]
    return(tf.concat(channels, 2))

def get_preprocessing():
    def preprocessing_fn(image, output_height=224, output_width=224):
        image = tf.expand_dims(image, 0)
        resized_image = tf.image.resize_bilinear(image, [output_height, output_width], align_corners=False)
        resized_image = tf.squeeze(resized_image)
        resized_image.set_shape([output_height, output_width, 3])
        image = tf.to_float(resized_image)
        return(mean_image_subtraction(image, [123.68, 116.78, 103.94]))
    return(preprocessing_fn)

def score_model(file_list):
    model_dir = os.path.join(cs.IMAGE_DIR, 'TFRetrainCheckpoints')
    results = []
    
    with tf.Graph().as_default():
        network_fn = get_network_fn(num_classes=6, is_training=False)
        image_preprocessing_fn = get_preprocessing()
        
        current_image = tf.placeholder(tf.uint8, shape=(224, 224, 3))
        preprocessed_image = image_preprocessing_fn(current_image, 224, 224)
        image  = tf.expand_dims(preprocessed_image, 0)
        logits, _ = network_fn(image)
        predictions = tf.argmax(logits, 1)
        
        with tf.Session() as sess:
            my_saver = tf.train.Saver()
            my_saver.restore(sess, tf.train.latest_checkpoint(model_dir))
            
            for file in file_list:
                imported_image_np = imread(file)
                result = sess.run(predictions, feed_dict={current_image: imported_image_np})
                true_label = get_nlcd_id(file)
                results.append([file, true_label, result[0]])

    return(results)

def get_nlcd_id(my_filename):
    ''' Extracts the true label  '''
    folder, _ = os.path.split(my_filename)
    return(cs.LABELS[os.path.basename(folder)])

#-----------------------------------
# Todo: make model smaller
def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    nn = tf.reshape(features, [-1, 224, 224, 3])

    # Convolutional Layer #1
    nn = tf.layers.conv2d(
        inputs=nn,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu
    )

    # Pooling Layer #1
    nn = tf.layers.max_pooling2d(inputs=nn, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    nn = tf.layers.conv2d(
        inputs=nn,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu
    )
    nn = tf.layers.max_pooling2d(inputs=nn, pool_size=[2, 2], strides=2)
    
    # Convolutional Layer #3 and Pooling Layer #3
    nn = tf.layers.conv2d(
        inputs=nn,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu
    )
    nn = tf.layers.max_pooling2d(inputs=nn, pool_size=[2, 2], strides=2)

    # Dense Layer
    nn = tf.contrib.layers.flatten(nn)
    nn = tf.layers.dense(inputs=nn, units=512, activation=tf.nn.relu)
    nn = tf.layers.dropout(inputs=nn, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=nn, units=10)

    loss = None
    train_op = None

    # Calculate Loss (for both TRAIN and EVAL modes)
    print(labels)
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=1e-4,
            optimizer='Adam'
        )

    # Generate Predictions
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # Return a ModelFnOps object
    return model_fn.ModelFnOps(mode=mode, predictions=predictions, loss=loss, train_op=train_op)


def load_data(image_info):
    labels = image_info[image_info.columns[2]].values
    print("Number of pictures: ", image_info.shape[0])
    data = np.empty((image_info.shape[0],224,224,3), np.float32)
    for index, row in image_info.iterrows():
        path = row.iloc[0]
        image = imread(path)
        image = np.expand_dims(image, axis=0)
        data[index, :] = np.array(image)
    print(data.shape)
    return data, labels


def merge_classes(y):
    # Regroup into three classes: Undeveloped, Cultivated, Developed. 6 is the new class label.
    y[y==cs.LABELS["Barren"]] = 6
    y[y==cs.LABELS["Forest"]] = 6
    y[y==cs.LABELS["Shrub"]] = 6
    y[y==cs.LABELS["Herbaceous"]] = 6
    return y