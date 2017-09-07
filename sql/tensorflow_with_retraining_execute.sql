USE [land_use_database]
GO

exec [dbo].[PreprocessingTF];
exec [dbo].[RetrainTF];
exec [dbo].[ScoreTF];
exec [dbo].[EvaluateTF];