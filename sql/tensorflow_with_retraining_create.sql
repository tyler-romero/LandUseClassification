-- TODO:
-- Right now the model reads and (attempts to) writes to the filetables. Without db_writers permissions for SQLRUserGroup, the writes will fail. Potential workaround: Write to a temporary folder, and then
--	Perform a manual selection and insertion into the filetable using the users own account. Problem with that is that the python package that attepts to locate the default temp directory (tempfile) doesnt always return the same directory
--	when executed from T-SQL.
-- Add arguements to Score and Evaluate to accomodate both validation and testing

USE [land_use_database]
GO


SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO


IF OBJECT_ID('[dbo].[PreprocessingTF]', 'P') IS NOT NULL  
    DROP PROCEDURE [dbo].[PreprocessingTF];  
GO  

CREATE PROCEDURE [dbo].[PreprocessingTF]
AS
BEGIN
	-- SET NOCOUNT ON added to prevent extra result sets from
	-- interfering with SELECT statements.
	SET NOCOUNT ON;

    -- Insert statements for procedure here
	EXECUTE sp_execute_external_script
      @language = N'Python'
    , @script = N'
import os
import numpy as np

import land_use.connection_settings as cs
import land_use.land_use_classification_utils as luc

repo_dir = cs.REPO_DIR
image_dir = cs.IMAGE_DIR
label_to_number_dict = cs.LABELS

np.random.seed(5318)

training_image_dir = os.path.join(cs.IMAGE_DIR, "TrainData")
training_filenames = luc.find_images(training_image_dir)
training_filenames = np.random.permutation(training_filenames)
df = luc.write_dataset("aerial", "train", training_filenames, training_image_dir, n_shards=50)
df.to_csv(os.path.join(training_image_dir, "dataset_split_info.csv"), index=False)

with open(os.path.join(training_image_dir, "labels.txt"), "w") as f:
    for key, value in label_to_number_dict.items():
        f.write("{}:{}\n".format(key, value))
';
    
END
GO


IF OBJECT_ID('[dbo].[RetrainTF]', 'P') IS NOT NULL  
    DROP PROCEDURE [dbo].[RetrainTF];  
GO  

CREATE PROCEDURE [dbo].[RetrainTF] 
AS
BEGIN
	-- SET NOCOUNT ON added to prevent extra result sets from
	-- interfering with SELECT statements.
	SET NOCOUNT ON;

	-- Insert statements for procedure here
	DECLARE @predictScript NVARCHAR(MAX);
	SET @predictScript = N'
import os
import numpy as np
import tempfile

from land_use.retrain import retrain
import land_use.connection_settings as cs
import land_use.land_use_classification_utils as luc

# path where retrained model and logs will be saved during training
train_data = os.path.join(cs.IMAGE_DIR, "TrainData")
train_dir = os.path.join(cs.IMAGE_DIR, "TFRetrainCheckpoints")
# temp_dir = tempfile.gettempdir()	# Cannot write to TFRetrainCheckpoints from Python (SQLRUserGroup)_), so write to a temp directory, then move files
# print("TempDir", temp_dir)

# Retrain the model. This can take hours.
retrain(train_data, train_dir)
	'
	EXECUTE sp_execute_external_script
	@language = N'python',
	@script = @predictScript;

END
GO


IF OBJECT_ID('[dbo].[ScoreTF]', 'P') IS NOT NULL  
    DROP PROCEDURE [dbo].[ScoreTF];  
GO 

CREATE PROCEDURE [dbo].[ScoreTF] 
AS
BEGIN
	-- SET NOCOUNT ON added to prevent extra result sets from
	-- interfering with SELECT statements.
	SET NOCOUNT ON;

	DROP TABLE IF EXISTS [dbo].[tf_val_predictions]
	CREATE TABLE [dbo].[tf_val_predictions](
		[filename] nvarchar(max),
		[true_label] int,
		[predicted_label] int
	)
	
	DECLARE @predictScript NVARCHAR(MAX);
	SET @predictScript = N'
import os
import glob
import numpy as np
import pandas as pd

import land_use.connection_settings as cs
import land_use.land_use_classification_utils as luc

repo_dir = cs.REPO_DIR
image_dir = cs.IMAGE_DIR
label_to_number_dict = cs.LABELS

dataset_dir = os.path.join(cs.IMAGE_DIR, "ValData")
file_list = glob.glob("{}/*/*.png".format(dataset_dir))	# TODO: get the file list using a filetable query

results_tf = luc.score_model(file_list)
print("Scored {} images".format(len(results_tf)))

tf_df = pd.DataFrame(results_tf, columns=["filename", "true_label", "predicted_label"])
OutputDataSet = tf_df
	'
	INSERT INTO [dbo].[tf_val_predictions] (filename, true_label, predicted_label)
	EXECUTE sp_execute_external_script
	@language = N'python',
	@script = @predictScript;

END
GO


IF OBJECT_ID('[dbo].[EvaluateTF]', 'P') IS NOT NULL  
    DROP PROCEDURE [dbo].[EvaluateTF];  
GO 

CREATE PROCEDURE [dbo].[EvaluateTF] 
AS
BEGIN
	-- SET NOCOUNT ON added to prevent extra result sets from
	-- interfering with SELECT statements.
	SET NOCOUNT ON;
	
	DECLARE @predictScript NVARCHAR(MAX);
	SET @predictScript = N'
import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import land_use.connection_settings as cs
import land_use.land_use_classification_utils as luc

repo_dir = cs.REPO_DIR
image_dir = cs.IMAGE_DIR
label_to_number_dict = cs.LABELS

tf_df = InputDataSet
y_pred = np.array(tf_df["predicted_label"])
y_true = np.array(tf_df["true_label"])

# Accuracy (all classes)
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy (all classes): ", accuracy)

# Accuracy (Undeveloped, Cultivated, Developed)
y_true = luc.merge_classes(y_true)
y_pred = luc.merge_classes(y_pred)

accuracy = accuracy_score(y_true, y_pred)
print("Accuracy (Undeveloped, Cultivated, Developed): ", accuracy)

	'
	EXECUTE sp_execute_external_script
	@language = N'python',
	@script = @predictScript,
	@input_data_1 = N'SELECT filename, true_label, predicted_label FROM dbo.tf_val_predictions';

END
GO