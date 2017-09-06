USE [land_use_database]
GO

exec [dbo].[GenerateFeaturesML] @input_table = N'TrainData', @output_table = N'train_features';
exec [dbo].[GenerateFeaturesML] @input_table = N'ValData', @output_table = N'validation_features';
exec [dbo].[GenerateFeaturesML] @input_table = N'TestData', @output_table = N'test_features';

exec [dbo].[TrainModelML];
exec [dbo].[ScoreModelML];
exec [dbo].[EvaluateModelML];