USE [land_use_database]
GO

exec [dbo].[TrainModelTF];
exec [dbo].[ScoreModelTF] @mode = 'val';
exec [dbo].[ScoreModelTF] @mode = 'test';