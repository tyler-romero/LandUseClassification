USE [land_use_database]
GO


SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO


IF OBJECT_ID('[dbo].[Preprocessing]', 'P') IS NOT NULL  
    DROP PROCEDURE [dbo].[Preprocessing];  
GO  

CREATE PROCEDURE [dbo].[Preprocessing]
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
import tempfile

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


IF OBJECT_ID('[dbo].[Retrain]', 'P') IS NOT NULL  
    DROP PROCEDURE [dbo].[Retrain];  
GO  

CREATE PROCEDURE [dbo].[Retrain] 
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

# Retrain the model. This can take hours.
temp_dir = tempfile.gettempdir()
print("TempDir", temp_dir)
#retrain(train_data, temp_dir)
	'
	EXECUTE sp_execute_external_script
	@language = N'python',
	@script = @predictScript;

END
GO


IF OBJECT_ID('[dbo].[ScoreModel]', 'P') IS NOT NULL  
    DROP PROCEDURE [dbo].[ScoreModel];  
GO  


IF OBJECT_ID('[dbo].[Score]', 'P') IS NOT NULL  
    DROP PROCEDURE [dbo].[Score];  
GO 

CREATE PROCEDURE [dbo].[Score] 
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

import land_use.connection_settings as cs
import land_use.land_use_classification_utils as luc

repo_dir = cs.REPO_DIR
image_dir = cs.IMAGE_DIR
label_to_number_dict = cs.LABELS

dataset_dir = os.path.join(cs.IMAGE_DIR, "ValData")
file_list = glob.glob("{}/*/*.png".format(dataset_dir))

start = pd.datetime.now()
results_tf = luc.score_model(file_list)
print("Scored {} images".format(len(results_tf)))
stop = pd.datetime.now()
print(stop - start)

tf_df = pd.DataFrame(results_tf, columns=["filename", "true_label", "predicted_label"])
# TODO: write this
	'
	EXECUTE sp_execute_external_script
	@language = N'python',
	@script = @predictScript;

END
GO


IF OBJECT_ID('[dbo].[Evaluate]', 'P') IS NOT NULL  
    DROP PROCEDURE [dbo].[Evaluate];  
GO 

CREATE PROCEDURE [dbo].[Evaluate] 
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

# TODO: Read tf_df
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
	@script = @predictScript;

END
GO