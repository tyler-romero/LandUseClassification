import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

import revoscalepy as revo
import microsoftml as ml

import utils.connection_settings as cs


def generate_csv(training_image_dir, label_to_number_dict):
    csv_path = os.path.join(training_image_dir, cs.LAND_USE_CSV)
    with open(csv_path, 'w') as map_file:
        map_file.write('image, label\n')
        for label in np.sort(os.listdir(training_image_dir)):
            my_dir = os.path.join(training_image_dir, label)
            if not os.path.isdir(my_dir):
                continue
            for filename in os.listdir(my_dir):
                map_file.write('{}, {}\n'.format(os.path.join(my_dir, filename), label_to_number_dict[label]))
    return csv_path


def compute_features(data, model, compute_context):
    revo.rx_data_step(data, "data.xdf", overwrite=True)
    data_xdf = revo.RxXdfData("data.xdf")
    featurized_data = ml.rx_featurize(
            data=data_xdf,                # Using xdf data as a workaround for a problem featurize was having with data frames
            # output_data=features_sql,   # The type (RxSqlServerData) for file is not supported. TODO: use RxSqlServerData for output when its supported
            overwrite=True,
            ml_transforms=[
                ml.load_image(cols={"feature": "image"}),
                ml.resize_image(cols="feature", width=224, height=224),
                ml.extract_pixels(cols="feature"),
                ml.featurize_image(cols="feature", dnn_model=model)
            ],
            ml_transform_vars=["path"],
            report_progress=2,
            verbose=2,
            compute_context=compute_context
        )
    featurized_data.columns = ["image", "label"] + ["f"+str(i) for i in range(len(featurized_data.columns)-2)]
    return featurized_data


def create_formula(data, is_dataframe=False):
    if is_dataframe:
        features_all = list(data.columns)
    else:
        features_all = revo.rx_get_var_names(data)
    features_to_remove = ["label", "image"]
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