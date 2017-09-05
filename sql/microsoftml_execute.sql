USE [land_use_database]
GO

exec [dbo].[GenerateFeatures] @input_table = N'TrainData', @output_table = N'train_features';
exec [dbo].[GenerateFeatures] @input_table = N'ValData', @output_table = N'validation_features';
exec [dbo].[GenerateFeatures] @input_table = N'TestData', @output_table = N'test_features';

exec [dbo].[TrainModel];
exec [dbo].[ScoreModel];
exec [dbo].[EvaluateModel];