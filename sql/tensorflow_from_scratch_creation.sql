-- TODO:
--	Right now the model attempts to load the entire dataset in memory. Transition to a batched approach, such as the TFRecords used in the retraining version of this scenario.
--	Note that the from scratch model performs less well than the transfer learning model when they are trained in the python notebook. Consider if it is even worth creating stored procedures for the from scratch version.

USE [land_use_database]
GO


SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO


IF OBJECT_ID('[dbo].[TrainModelTF]', 'P') IS NOT NULL  
    DROP PROCEDURE [dbo].[TrainModelTF];  
GO  

CREATE PROCEDURE [dbo].[TrainModelTF] 
AS
BEGIN
	-- SET NOCOUNT ON added to prevent extra result sets from
	-- interfering with SELECT statements.
	SET NOCOUNT ON;

	-- Insert statements for procedure here
	DECLARE @predictScript NVARCHAR(MAX);
	SET @predictScript = N'
# Imports
import os, sys
import numpy as np
import tensorflow as tf
import pandas as pd

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn

import land_use.connection_settings as cs
import land_use.land_use_classification_utils as luc

image_dir = cs.IMAGE_DIR
tf_model_dir = os.path.join(cs.IMAGE_DIR, "TFTrainCheckpoints")
connection_string = cs.get_connection_string()
label_to_number_dict = cs.LABELS

tf.logging.set_verbosity(tf.logging.INFO)

# Load Training Data
train_info = luc.gather_image_paths(image_dir, label_to_number_dict, connection_string, mode="train")	# Todo: get paths using input_data. Be careful of non .png files
train_data, train_labels = luc.load_data(train_info)

print(train_data.shape)
print(train_labels.shape)

# Create the Estimator
classifier = learn.SKCompat(learn.Estimator(model_fn=luc.cnn_model_fn, model_dir=tf_model_dir, config=learn.RunConfig(keep_checkpoint_max=1)))

# Train the model
classifier.fit(
    x=train_data,
    y=train_labels,
    batch_size=100,
    steps=5000
)
	'
	EXECUTE sp_execute_external_script
	@language = N'python',
	@script = @predictScript;

END
GO


IF OBJECT_ID('[dbo].[ScoreModelTF]', 'P') IS NOT NULL  
    DROP PROCEDURE [dbo].[ScoreModelTF];  
GO 

CREATE PROCEDURE [dbo].[ScoreModelTF] @mode nvarchar(4)
AS
BEGIN
	-- SET NOCOUNT ON added to prevent extra result sets from
	-- interfering with SELECT statements.
	SET NOCOUNT ON;
	
	DECLARE @predictScript NVARCHAR(MAX);
	SET @predictScript = N'
# Imports
import os, sys
import numpy as np
import tensorflow as tf
import pandas as pd

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn

import land_use.connection_settings as cs
import land_use.land_use_classification_utils as luc

image_dir = cs.IMAGE_DIR
tf_model_dir = os.path.join(cs.IMAGE_DIR, "TFTrainCheckpoints")
connection_string = cs.get_connection_string()
label_to_number_dict = cs.LABELS

tf.logging.set_verbosity(tf.logging.INFO)

# Load Data
info = luc.gather_image_paths(image_dir, label_to_number_dict, connection_string, mode=mode)
data, labels = luc.load_data(info)

print(data.shape)
print(labels.shape)

# Configure the accuracy metric for evaluation
metrics = {
    "accuracy": learn.MetricSpec(metric_fn=tf.metrics.accuracy, prediction_key="classes"),
}

# Create the Estimator
classifier = learn.SKCompat(learn.Estimator(model_fn=luc.cnn_model_fn, model_dir=tf_model_dir, config=learn.RunConfig(keep_checkpoint_max=1)))

# Evaluate the model and print results
results = classifier.score(
    x=data,
    y=labels,
    metrics=metrics
)
print(results)
	'
	EXECUTE sp_execute_external_script
	@language = N'python',
	@script = @predictScript,
	@params = N'@mode varchar(4)',
	@mode = @mode;

END
GO