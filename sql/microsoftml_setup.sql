USE [land_use_database]
GO


SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO


IF OBJECT_ID('[dbo].[GenerateFeaturesML]', 'P') IS NOT NULL  
    DROP PROCEDURE [dbo].GenerateFeaturesML;  
GO  

CREATE PROCEDURE [dbo].GenerateFeaturesML @input_table varchar(max), @output_table varchar(max)
AS
BEGIN
	-- SET NOCOUNT ON added to prevent extra result sets from
	-- interfering with SELECT statements.
	SET NOCOUNT ON;

	DECLARE @root nvarchar(1000) = FileTableRootPath(); 

	DECLARE @batchPatientImages nvarchar(max)
	SET @batchPatientImages = 
		N'SELECT (''' + @root + ''' + [file_stream].GetFileNamespacePath()) as image
			FROM [' + @input_table + '] WHERE [is_directory] = 0';
	print @batchPatientImages;

    -- Insert statements for procedure here
	EXECUTE sp_execute_external_script
      @language = N'Python'
    , @script = N'
import revoscalepy as revo
import microsoftml as ml
import inspect

import land_use.connection_settings as cs
import land_use.land_use_classification_utils as luc
print(inspect.getfile(cs))

label_to_number_dict = cs.LABELS
connection_string = cs.get_connection_string()

data = InputDataSet
luc.gather_image_paths2(data, label_to_number_dict)
print("Data: ", data.shape)

luc.compute_features(data, output_table, connection_string)
'
	, @input_data_1 = @batchPatientImages
	, @params = N' @output_table varchar(50)'
	, @output_table = @output_table;
    
	
END
GO


IF OBJECT_ID('[dbo].[TrainModelML]', 'P') IS NOT NULL  
    DROP PROCEDURE [dbo].[TrainModelML];  
GO  

CREATE PROCEDURE [dbo].[TrainModelML] 
AS
BEGIN
	-- SET NOCOUNT ON added to prevent extra result sets from
	-- interfering with SELECT statements.
	SET NOCOUNT ON;

	-- Insert statements for procedure here
	DECLARE @predictScript NVARCHAR(MAX);
	SET @predictScript = N'
import os, sys
import numpy as np
import pandas as pd
import revoscalepy as revo
import microsoftml as ml

import land_use.connection_settings as cs
import land_use.land_use_classification_utils as luc

image_dir = cs.IMAGE_DIR
label_to_number_dict = cs.LABELS

connection_string = cs.get_connection_string()

# Set recursion limit to be slightly larger to accommodate larger formulas (which are paresed recursively)
print("Old recursion limit: ", sys.getrecursionlimit())
sys.setrecursionlimit(1500)
print("New recursion limit: ", sys.getrecursionlimit())

train_features_sql = revo.RxSqlServerData(table=cs.TABLE_TRAIN_FEATURES, connection_string=connection_string)

formula = luc.create_formula(train_features_sql, is_dataframe=False)
print(formula)

model = ml.rx_neural_network(formula=formula,
                            data=train_features_sql,
                            num_hidden_nodes=100,
                            num_iterations=100,
                            max_norm = 0,
                            init_wts_diameter=0.1,
                            mini_batch_size=10,
                            method="multiClass",
                            verbose=2)
luc.insert_model(cs.TABLE_MODELS, connection_string, model, "rx_neural_network")
	'
	EXECUTE sp_execute_external_script
	@language = N'python',
	@script = @predictScript;

END
GO


IF OBJECT_ID('[dbo].[ScoreModelML]', 'P') IS NOT NULL  
    DROP PROCEDURE [dbo].[ScoreModelML];  
GO  

CREATE PROCEDURE [dbo].[ScoreModelML] 
AS
BEGIN
	-- SET NOCOUNT ON added to prevent extra result sets from
	-- interfering with SELECT statements.
	SET NOCOUNT ON;

	DECLARE @predictScript NVARCHAR(MAX);
	SET @predictScript = N'
import revoscalepy as revo
import microsoftml as ml

import land_use.connection_settings as cs
import land_use.land_use_classification_utils as luc

image_dir = cs.IMAGE_DIR
label_to_number_dict = cs.LABELS

# Connect to SQL Server and set compute context
connection_string = cs.get_connection_string()

model = luc.retrieve_model(cs.TABLE_MODELS, connection_string, "rx_neural_network")

val_features_sql = revo.RxSqlServerData(table=cs.TABLE_VALIDATION_FEATURES, connection_string=connection_string)
predictions = ml.rx_predict(model, data=val_features_sql, extra_vars_to_write=["label", "image"])

number_to_label_dict = {}
for label, number in label_to_number_dict.items():
    number_to_label_dict["Score."+str(number)] = label
print(number_to_label_dict)
    
predictions=predictions.rename(columns = number_to_label_dict)

predictions_sql = revo.RxSqlServerData(table=cs.TABLE_VAL_PREDICTIONS, connection_string=connection_string)
revo.rx_data_step(predictions, predictions_sql, overwrite=True)
print(predictions.head())

	'
	EXECUTE sp_execute_external_script
	@language = N'python',
	@script = @predictScript;

END
GO


IF OBJECT_ID('[dbo].[EvaluateModel]', 'P') IS NOT NULL  
    DROP PROCEDURE [dbo].[EvaluateModelML];  
GO 

CREATE PROCEDURE [dbo].[EvaluateModelML] 
AS
BEGIN
	-- SET NOCOUNT ON added to prevent extra result sets from
	-- interfering with SELECT statements.
	SET NOCOUNT ON;
	
	DECLARE @predictScript NVARCHAR(MAX);
	SET @predictScript = N'
import os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import revoscalepy as revo
import microsoftml as ml

import land_use.connection_settings as cs
import land_use.land_use_classification_utils as luc

image_dir = cs.IMAGE_DIR
label_to_number_dict = cs.LABELS

# Connect to SQL Server and set compute context
connection_string = cs.get_connection_string()

model = luc.retrieve_model(cs.TABLE_MODELS, connection_string, "rx_neural_network")

test_features_sql = revo.RxSqlServerData(table=cs.TABLE_TEST_FEATURES, connection_string=connection_string)
predictions = ml.rx_predict(model, data=test_features_sql, extra_vars_to_write=["label", "image"])

number_to_label_dict = {}
for label, number in label_to_number_dict.items():
    number_to_label_dict["Score."+str(number)] = label
print(number_to_label_dict)

predictions=predictions.rename(columns = number_to_label_dict)

predictions_sql = revo.RxSqlServerData(table=cs.TABLE_TEST_PREDICTIONS, connection_string=connection_string)
revo.rx_data_step(predictions, predictions_sql, overwrite=True)
print(predictions.head())

class_probs = np.array(predictions.drop(["label", "image"], axis=1))
y_pred = np.argmax(class_probs, axis=1)
y_true = np.array(predictions["label"])

# Accuracy (all classes)
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy (all classes): ", accuracy)

# Accuracy (Undeveloped, Cultivated, Developed)
# Let 6 be a new label indicating undeveloped land as a whole
y_true[y_true==label_to_number_dict["Barren"]] = 6
y_true[y_true==label_to_number_dict["Forest"]] = 6
y_true[y_true==label_to_number_dict["Shrub"]] = 6
y_true[y_true==label_to_number_dict["Herbaceous"]] = 6

y_pred[y_pred==label_to_number_dict["Barren"]] = 6
y_pred[y_pred==label_to_number_dict["Forest"]] = 6
y_pred[y_pred==label_to_number_dict["Shrub"]] = 6
y_pred[y_pred==label_to_number_dict["Herbaceous"]] = 6

accuracy = accuracy_score(y_true, y_pred)
print("Accuracy (Undeveloped, Cultivated, Developed): ", accuracy)
	'
	EXECUTE sp_execute_external_script
	@language = N'python',
	@script = @predictScript;

END
GO