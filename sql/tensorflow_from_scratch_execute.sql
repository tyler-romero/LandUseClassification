USE [land_use_database]
GO

--exec [dbo].[Train];
exec [dbo].[Score] @mode = 'val';
--exec [dbo].[Score] @mode = 'test';