# Server constants
DB_DRIVER = 'ODBC Driver 13 for SQL Server'
#SERVER_NAME = 'localhost'
SERVER_NAME = 'TYLER-LAPTOP\TYLERSQLSERVER'
#PORT_NUMBER = '1443'
PORT_NUMBER = '21816'
DATABASE_NAME = 'land_use_database'
USERID = 'demo'
PASSWORD = 'D@tascience'

# Path to images
TRAINING_IMAGE_DIR = 'C:\\Users\\t-tyrome\\Documents\\Internship\\LandUseClassification\\data\\balanced_training_set'
VALIDATION_IMAGE_DIR = 'C:\\Users\\t-tyrome\\Documents\\Internship\\LandUseClassification\\data\\balanced_validation_set'
TEST_IMAGE_DIR = 'C:\\Users\\t-tyrome\\Documents\\Internship\\LandUseClassification\\data\\balanced_test_set'

#Tables
TABLE_TRAIN_FEATURES = "dbo.train_features"
TABLE_VALIDATION_FEATURES = "dbo.validation_features"
TABLE_TEST_FEATURES = "dbo.validation_features"
TABLE_MODELS = "dbo.models"
TABLE_VAL_PREDICTIONS = "dbo.val_predictions"
TABLE_TEST_PREDICTIONS = "dbo.test_predictions"

#Functions
def get_connection_string():
	driver = 'DRIVER={' + DB_DRIVER + '}'
	port = 'PORT=' + PORT_NUMBER
	server = 'SERVER=' + SERVER_NAME
	database = 'DATABASE=' + DATABASE_NAME
	uid = 'UID=' + USERID
	pwd = 'PWD=' + PASSWORD
	connection_string = ';'.join([driver,server,port,database,uid,pwd])
	print(connection_string)
	return connection_string

# Variables
MICROSOFTML_MODEL_NAME = "Resnet18"
FASTTREE_MODEL_NAME = "rx_fast_trees"
LAND_USE_CSV = "land_use.csv"