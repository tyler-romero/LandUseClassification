-- Follow the instruction in readme file to enable FileTable on your database

-- Conigure new database  
CREATE DATABASE land_use_database  
WITH FILESTREAM ( NON_TRANSACTED_ACCESS = FULL, DIRECTORY_NAME = N'FileTableData' )

-- Configure existing database
ALTER DATABASE land_use_database  
SET FILESTREAM ( NON_TRANSACTED_ACCESS = FULL, DIRECTORY_NAME = N'FileTableData' )

-- Creating a FileTable table
---- If you encounter the error: "Default FILESTREAM filegroup is not available in database 'land_use_database', then
---- follow instructions here: http://www.kodyaz.com/t-sql/default-filestream-filegroup-is-not-available-in-database.aspx
USE land_use_database
GO
CREATE TABLE TrainData AS FileTable  
WITH (   
	FileTable_Directory = 'TrainData',  
	FileTable_Collate_Filename = database_default  
);

CREATE TABLE ValData AS FileTable  
WITH (   
	FileTable_Directory = 'ValData',  
	FileTable_Collate_Filename = database_default  
);

CREATE TABLE TestData AS FileTable  
WITH (   
	FileTable_Directory = 'TestData',  
	FileTable_Collate_Filename = database_default  
);  

CREATE TABLE TFPretrainedModels AS FileTable  
WITH (   
	FileTable_Directory = 'TFPretrainedModels',  
	FileTable_Collate_Filename = database_default  
);  

CREATE TABLE TFRetrainCheckpoints AS FileTable  
WITH (   
	FileTable_Directory = 'TFRetrainCheckpoints',  
	FileTable_Collate_Filename = database_default  
);

CREATE TABLE TFTrainCheckpoints AS FileTable  
WITH (   
	FileTable_Directory = 'TFTrainCheckpoints',  
	FileTable_Collate_Filename = database_default  
);  